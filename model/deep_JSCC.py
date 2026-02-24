import torch
import torch.nn as nn
from einops import rearrange
from model.layers import *
import math
from compressai.layers import conv1x1

__all__ = [
        #'DeepJSCC','DeepJSCC_single','DeepJSCC_multi', 
        'ResNet', 'ResNet_single', 'ResNet_multi',
        'ConvNext', 'ConvNext_single', 'ConvNext_multi']

# ==========================================
# 1. Base Class (Common Logic & Parser)
# ==========================================
class JSCC_Common(nn.Module):
    def __init__(self, dim, spp, SNRdB, **kwargs):
        super().__init__()
        self.ds_factor = 2**4 # Downsample factor (16)
        self.spp = spp
        self.snr = SNRdB
        self.dim = dim
        # 128이잖아. 5로 grouped convolution을 하나? PatchMerge랑 PatchReverse 코드 수정해서 달성.
        raw_symbol_dim = int(spp * (self.ds_factor**2) * 2)
        self.symbol_dim = round(raw_symbol_dim/5)*5
        
        self.chunk_size=round(raw_symbol_dim/5)
        
        # F: Max number of chunks
        self.F = 5
        self.min_chunks = 1
        
        self.channel = AWGNChannel(SNRdB)
        self.eval_chunk = None

    @staticmethod
    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.patch_shape', type=tuple)
        parser.add_argument('--model.window_size', type=int) 
        parser.add_argument('--model.head_dims', type=tuple) 
        return parser

    def _get_inst_csi(self, B, device):
        if self.training:
            return torch.randint(-5, 25, (B, 1), device=device).float(), torch.randint(self.min_chunks, self.F, (B, 1), device=device).float()
        else:
            return torch.ones(B, 1, device=device) * self.snr, torch.ones(B, 1, device=device) * (self.eval_chunk if self.eval_chunk is not None else self.F)

    def _simulate_channel(self, y, H, W):
        # y: (B*n, symbol_dim, H, W) or equivalent
        y_complex = rearrange(y, "b (n_c iq) h w -> b (n_c h w) iq", iq=2)
        y_complex = torch.complex(y_complex[..., 0], y_complex[..., 1])
        
        # Power Normalization
        power = torch.mean(torch.abs(y_complex)**2,dim=1,keepdim=True) + 1e-6
        y_complex = y_complex / torch.sqrt(power)
        
        # Channel
        y_hat_complex = self.channel(y_complex, snr_db=self.instsnr)

        # Back to Real
        y_hat = torch.stack((y_hat_complex.real, y_hat_complex.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)
        return y_hat, y
    
    def loss(self, out_net:dict, target:torch.Tensor):
        x_hat=out_net['x_net']
        B, F, C, H, W = x_hat.shape
        weights=torch.tensor([i for i in range(self.min_chunks,self.F+1)], dtype=torch.float32).to(x_hat.device)
        weights=weights/weights.mean(dim=0)
        target_rep = target.unsqueeze(1).expand(-1, F, -1, -1, -1)  # (B, F, 3, H, W)
        # --- per-step MSE: average over C, H, W, then batch ---------------
        mse_curve = torch.mean((x_hat - target_rep) ** 2, dim=(0, 2, 3, 4))   # (F)
        out={
            "MSE"           : mse_curve[-1],
        }
        for i in range(self.min_chunks, self.F+1,max(1, (self.F+1-self.min_chunks)//4)):
            out[f"MSE(~{i+1})"]=mse_curve[i]
        # --- Loss: only the first reconstruction drives back-prop ---------
        out['loss'] = torch.mean(weights*mse_curve)
        
        return out

# ==========================================
# 2. Sequential Base (For Base & Single)
# ==========================================
class JSCC_Sequential(JSCC_Common):
    def forward(self, x):
        B, _, H, W = x.shape
        self.instsnr, self.instchunk = self._get_inst_csi(B, x.device)

        # Encoding
        y = self.encoder(x)
        y=y.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        _, _, H, W = y.shape

        # Channel
        y_hat, y = self._simulate_channel(y, H, W)
        
        # Bandwidth Adaptation (Masking)
        ys = []
        for i in range(self.min_chunks, self.F + 1):
            mask = torch.zeros((B, self.symbol_dim, 1, 1), device=x.device)
            mask[:, :self.chunk_size*i, :, :] = 1.0
            ys.append(y_hat * mask)
        
        y_hats = torch.cat(ys, dim=0)

        y_hats=y_hats.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # Decoding
        x_hats = self.decoder(y_hats)
        x_hats = torch.clamp(x_hats, 0, 1)
        x_hat = torch.split(x_hats, B, dim=0)

        return {"x_hat": x_hat, "y": y}

# ==========================================
# 3. Multi Base (For Multi)
# ==========================================
class JSCC_Multi_Base(JSCC_Common):
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        self.instsnr, self.instchunk  = self._get_inst_csi(B, x.device)
        
        # Encoding Loop
        for i in range(0, 4):
            x = self.ds_layer[i](x)
            x=torch.cat([self.enc_layer[i](x[:B]),x], dim=0)
        x=rearrange(x,'(f b) h w c -> b h w (f c)', b=B)
        y = self.projector(x)
        y=y.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        _, _, H, W = y.shape

        # Channel
        y_hat, y = self._simulate_channel(y, H, W)
        
        # Bandwidth Adaptation
        ys = []
        for i in range(self.min_chunks, self.F + 1):
            mask = torch.zeros((B, self.symbol_dim, 1, 1), device=x.device)
            mask[:, :self.chunk_size*i, :, :] = 1.0
            ys.append(self.dejector((y_hat * mask).permute(0, 2, 3, 1)))# (N, C, H, W) -> (N, H, W, C)

        y_hats = torch.stack(ys, dim=0)

        x_hats = []
        for y_hat in y_hats: 
            y_hat=rearrange(y_hat,'b h w (f c) -> (f b) h w c', c=self.dim)
            # Decoding Loop
            for i in range(0, 4):
                temp = self.dec_layer[i](rearrange(y_hat[:2*B], '(f b) h w c -> b h w (f c)', b=B))
                y_hat = torch.cat([temp,y_hat[2*B:]], dim=0)
                y_hat = self.us_layer[i](y_hat)
            x_hats.append(torch.clamp(y_hat, 0, 1))

        return {"x_hat": x_hats, "y": y}

# ==========================================
# 5. Implementations: ConvNext
# ==========================================
class ConvNext(JSCC_Sequential):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)

        self.encoder = nn.Sequential(
            PatchEmbedding(dim,2),
            NeXtBlock(dim), NeXtBlock(dim),
            PatchMerging(dim, 2*dim),
            NeXtBlock(2*dim), NeXtBlock(2*dim),
            PatchMerging(2*dim, 4*dim),
            NeXtBlock(4*dim), NeXtBlock(4*dim), NeXtBlock(4*dim),
            NeXtBlock(4*dim), NeXtBlock(4*dim), NeXtBlock(4*dim),
            PatchMerging(4*dim, 8*dim),
            NeXtBlock(8*dim), NeXtBlock(8*dim),
            nn.Linear(8*dim,self.symbol_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.symbol_dim,8*dim),
            NeXtBlock(8*dim), NeXtBlock(8*dim),
            PatchReverse(8*dim, 4*dim),
            NeXtBlock(4*dim), NeXtBlock(4*dim), NeXtBlock(4*dim),
            NeXtBlock(4*dim), NeXtBlock(4*dim), NeXtBlock(4*dim),
            PatchReverse(4*dim, 2*dim),
            NeXtBlock(2*dim), NeXtBlock(2*dim),
            PatchReverse(2*dim, dim),
            NeXtBlock(dim), NeXtBlock(dim),
            PatchReconstruct(dim, 2),
        )

class ConvNext_single(JSCC_Sequential):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)
        
        self.encoder = nn.Sequential(
            PatchEmbedding(dim,2),
            NeXtBlock(dim), NeXtBlock(dim), PatchMerging(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim), PatchMerging(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim), NeXtBlock(dim),
            NeXtBlock(dim), NeXtBlock(dim), NeXtBlock(dim),
            PatchMerging(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim),
            nn.Linear(dim,self.symbol_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.symbol_dim,dim),
            NeXtBlock(dim), NeXtBlock(dim), PatchReverse(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim), NeXtBlock(dim),
            NeXtBlock(dim), NeXtBlock(dim), NeXtBlock(dim),
            PatchReverse(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim), PatchReverse(dim, dim),
            NeXtBlock(dim), NeXtBlock(dim),
            PatchReconstruct(dim, 2),
        )

class ConvNext_multi(JSCC_Multi_Base):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)
        
        depth=[2,2,6,2]
        self.ds_layer = nn.ModuleList([
            PatchEmbedding(dim,2),
            PatchMerging(dim,dim),
            PatchMerging(dim,dim),
            PatchMerging(dim,dim),
        ])
        self.enc_layer = nn.ModuleList([
            nn.Sequential(*[NeXtBlock(dim) for _ in range(i)])
              for i in depth
        ])
        self.projector = nn.Sequential(nn.LayerNorm(5*dim),nn.Linear(5*dim,self.symbol_dim))
        
        depth.reverse()
        self.dejector = nn.Sequential(nn.Linear(self.symbol_dim, 5*dim),nn.LayerNorm(5*dim))
        self.dec_layer = nn.ModuleList([
            nn.Sequential(nn.Linear(2*dim,dim),*[NeXtBlock(dim) for _ in range(i)])
              for i in depth
        ])
        self.us_layer = nn.ModuleList([
            PatchReverse(dim,dim),
            PatchReverse(dim,dim),
            PatchReverse(dim,dim),
            PatchReconstruct(dim,2)
        ])

# ==========================================
# 6. Implementations: ResNet
# ==========================================
class ResNet(JSCC_Sequential):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)
        
        self.patchembed = PatchEmbedding(dim,2)
        self.sideinfo_proj = nn.Linear(2, n_u)
        self.enproj = conv1x1(dim + n_u, dim)
        self.deproj = nn.Linear(self.symbol_dim,8*dim)
        self.enproj2 = conv1x1(8*dim + n_u, 8*dim)
        
        self.encoder = nn.Sequential(
            PatchEmbedding(dim,2),
            resNetBlock(dim), resNetBlock(dim), PatchMerging(dim, 2*dim),
            resNetBlock(2*dim), resNetBlock(2*dim), PatchMerging(2*dim, 4*dim),
            resNetBlock(4*dim), resNetBlock(4*dim), resNetBlock(4*dim),
            resNetBlock(4*dim), resNetBlock(4*dim), resNetBlock(4*dim),
            PatchMerging(4*dim, 8*dim),
            resNetBlock(8*dim), resNetBlock(8*dim),
            nn.Linear(8*dim,self.symbol_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.symbol_dim,8*dim),
            resNetBlock(8*dim), resNetBlock(8*dim), PatchReverse(8*dim, 4*dim),
            resNetBlock(4*dim), resNetBlock(4*dim), resNetBlock(4*dim),
            resNetBlock(4*dim), resNetBlock(4*dim), resNetBlock(4*dim),
            PatchReverse(4*dim, 2*dim),
            resNetBlock(2*dim), resNetBlock(2*dim), PatchReverse(2*dim, dim),
            resNetBlock(dim), resNetBlock(dim),
            PatchReconstruct(dim, 2),
        )

class ResNet_single(JSCC_Sequential):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)
        
        self.encoder = nn.Sequential(
            PatchEmbedding(dim,2),
            resNetBlock(dim), resNetBlock(dim), PatchMerging(dim, dim),
            resNetBlock(dim), resNetBlock(dim), PatchMerging(dim, dim),
            resNetBlock(dim), resNetBlock(dim), resNetBlock(dim),
            resNetBlock(dim), resNetBlock(dim), resNetBlock(dim),
            PatchMerging(dim, dim),
            resNetBlock(dim), resNetBlock(dim),
            nn.Linear(dim,self.symbol_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.symbol_dim,dim),
            resNetBlock(dim), resNetBlock(dim), PatchReverse(dim, dim),
            resNetBlock(dim), resNetBlock(dim), resNetBlock(dim),
            resNetBlock(dim), resNetBlock(dim), resNetBlock(dim),
            PatchReverse(dim, dim),
            resNetBlock(dim), resNetBlock(dim), PatchReverse(dim, dim),
            resNetBlock(dim), resNetBlock(dim),
            PatchReconstruct(dim, 2),
        )

class ResNet_multi(JSCC_Multi_Base):
    def __init__(self, dim, spp, SNRdB, n_u=6, **kwargs):
        super().__init__(dim, spp, SNRdB, **kwargs)
        
        self.sideinfo_proj = nn.Linear(2, n_u)
        self.enproj = conv1x1(dim + n_u, dim)
        self.enproj2 = conv1x1(dim + n_u, dim)
        
        depth=[2,2,6,2]
        self.ds_layer = nn.ModuleList([
            PatchEmbedding(dim,2),
            PatchMerging(dim,dim),
            PatchMerging(dim,dim),
            PatchMerging(dim,dim),
        ])
        self.enc_layer = nn.ModuleList([
            nn.Sequential(*[resNetBlock(dim) for _ in range(i)])
              for i in depth
        ])
        self.projector = nn.Sequential(nn.LayerNorm(5*dim),nn.Linear(5*dim,self.symbol_dim))
        
        depth.reverse()
        self.dejector = nn.Sequential(nn.Linear(self.symbol_dim, 5*dim),nn.LayerNorm(5*dim))
        self.dec_layer = nn.ModuleList([
            nn.Sequential(nn.Linear(2*dim,dim),*[resNetBlock(dim) for _ in range(i)])
              for i in depth
        ])
        self.us_layer = nn.ModuleList([
            PatchReverse(dim,dim),
            PatchReverse(dim,dim),
            PatchReverse(dim,dim),
            PatchReconstruct(dim,2)
        ])