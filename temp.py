import torch
import torch.nn as nn
from compressai.layers import conv1x1, conv3x3
from einops import rearrange

class AWGNChannel(nn.Module):
    '''
    Trainable/Untrainable SISO Channel for torch
    Now supports batch-wise SNR
    '''
    def __init__(self, SNRdB=None):
        super(AWGNChannel, self).__init__()
        # Initialize with a default value if needed, though forward input is preferred
        self.default_snr = 10 ** (SNRdB / 10) if SNRdB is not None else None

    def forward(self, x, snr_db=None):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"
        
        # Power Normalization (Unit Average Power)
        # Normalize per element (or per channel/batch depending on requirement, usually per element for AWGN)
        pwr = torch.mean(torch.abs(x)**2, dim=-1, keepdim=True)
        x = x / torch.sqrt(pwr + 1e-9)

        # Noise Generation
        if snr_db is None:
            assert self.default_snr is not None, "SNR must be provided"
            snr_linear = self.default_snr
        else:
            # snr_db shape: (B, 1), 10^(SNR/10)
            snr_linear = 10 ** (snr_db / 10.0)
        noise_std = (1.0 / snr_linear) ** 0.5
            
        n = noise_std * torch.randn(x.shape, dtype=torch.complex64, device=x.device)
        
        # Add noise
        y = x + n
        return y

class resNetBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch,in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # (N, H, W, C)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out
    
class ResNet_baseline(nn.Module):
    def __init__(self, dim, spp, SNRdB):
        super().__init__()
        self.ds_factor=2**4
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(spp*(self.ds_factor**2)*2)
        self.symbol_dim=round(rwa_symbol_dim/4)*4
        self.spp=self.symbol_dim/(2*self.ds_factor**2)
        self.F=4 # max number of chunks, spp approximately 0.5
        self.chunk_size=round(rwa_symbol_dim/4)
        self.min_F=1 # minimum number of chunks, spp approximately 0.1

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.encoder=nn.Sequential(
            nn.Conv2d(dim+8,dim,1),
            resNetBlock(dim),resNetBlock(dim),
            nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=3,stride=2,padding=1),
            resNetBlock(2*dim),resNetBlock(2*dim),
            nn.Conv2d(in_channels=2*dim,out_channels=4*dim,kernel_size=3,stride=2,padding=1),
            resNetBlock(4*dim),resNetBlock(4*dim),
            nn.Conv2d(in_channels=4*dim,out_channels=self.symbol_dim,kernel_size=1, groups=4)
            )

        self.channel=AWGNChannel(SNRdB)

        self.decoder=nn.Sequential(
            nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=4*dim,kernel_size=1),
            resNetBlock(4*dim),resNetBlock(4*dim),
            nn.ConvTranspose2d(in_channels=4*dim,out_channels=2*dim,kernel_size=3,stride=2,padding=1,output_padding=1),
            resNetBlock(2*dim),resNetBlock(2*dim),
            nn.ConvTranspose2d(in_channels=2*dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1),
            resNetBlock(dim),resNetBlock(dim),
            nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)
        )
        
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)
        
        x1=self.ds_1(x)
        _,_,H1,W1=x1.shape
        latent=self.encoder(torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1))
        B,_,H,W=latent.shape

        latent=latent[:,:self.chunk_size*chunk_num]
        y=rearrange(latent, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)
        y=torch.complex(y[...,0],y[...,1])
        power=torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)

        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)

        # decoding
        noisy_latent=torch.zeros((B,self.symbol_dim,H,W),device=x.device)
        noisy_latent[:,:y_hat.shape[1]]=y_hat
        x_hat=self.decoder(torch.cat((noisy_latent,sideinfo.expand(-1,-1,H,W)),dim=1))
        
        return x_hat

class ResNet_multi1(nn.Module):
    def __init__(self, dim, spp, SNRdB):
        super().__init__()
        self.ds_factor=2**4
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(spp*(self.ds_factor**2)*2)
        self.symbol_dim=round(rwa_symbol_dim/4)*4
        self.spp=self.symbol_dim/(2*self.ds_factor**2)
        self.F=4 # max number of chunks, spp approximately 0.5
        self.chunk_size=round(rwa_symbol_dim/4)
        self.min_F=1 # minimum number of chunks, spp approximately 0.1

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_1=nn.Sequential(nn.Conv2d(dim+8,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.ds_2=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_2=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.ds_3=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_3=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.projector=nn.Conv2d(in_channels=4*dim,out_channels=self.symbol_dim,kernel_size=1, groups=4)

        self.channel=AWGNChannel(SNRdB)
        
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=4*dim,kernel_size=1, groups=4)
        self.dec_3=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_3=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_2=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_2=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_1=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_1=nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)
        
        x1=self.ds_1(x)
        _,_,H1,W1=x1.shape
        x2=self.enc_1(torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1))
        x2=self.ds_2(x2)
        _,_,H2,W2=x2.shape
        x3=self.enc_2(x2)
        x3=self.ds_3(x3)
        _,_,H3,W3=x3.shape
        x4=self.enc_3(x3)

        # Resolution sync
        x1=torch.nn.functional.adaptive_avg_pool2d(x1, (H3,W3))
        x2=torch.nn.functional.adaptive_avg_pool2d(x2, (H3,W3))
        x3=torch.nn.functional.adaptive_avg_pool2d(x3, (H3,W3))
        total=torch.cat((x4,x3,x2,x1),dim=1)

        # Sending
        latent=self.projector(total)
        latent=latent[:,:self.chunk_size*chunk_num]
        y=rearrange(latent, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)
        y=torch.complex(y[...,0],y[...,1])
        power=torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)

        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H3, w=W3)

        # decoding
        noisy_latent=torch.zeros((B,self.symbol_dim,H3,W3),device=x.device)
        noisy_latent[:,:y_hat.shape[1]]=y_hat
        noisy_latent=self.deprojector(torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1))
        y4,y3,y2,y1=torch.split(noisy_latent,self.dim, dim=1)
        
        y3=torch.nn.functional.interpolate(y3,(H3,W3),mode='bilinear')
        y2=torch.nn.functional.interpolate(y2,(H2,W2),mode='bilinear')
        y1=torch.nn.functional.interpolate(y1,(H1,W1),mode='bilinear')
        
        y3=self.dec_3(torch.cat((y4,y3),dim=1))
        y3=self.us_3(y3)
        y2=self.dec_2(torch.cat((y3,y2), dim=1))
        y2=self.us_2(y2)
        y1=self.dec_1(torch.cat((y2,y1), dim=1))
        x_hat=self.us_1(y1)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat

class ResNet_multi1group(nn.Module):
    def __init__(self, dim, spp, SNRdB):
        super().__init__()
        self.ds_factor=2**4
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(spp*(self.ds_factor**2)*2)
        self.symbol_dim=round(rwa_symbol_dim/4)*4
        self.spp=self.symbol_dim/(2*self.ds_factor**2)
        self.F=4 # max number of chunks, spp approximately 0.5
        self.chunk_size=round(rwa_symbol_dim/4)
        self.min_F=1 # minimum number of chunks, spp approximately 0.1

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_1=nn.Sequential(nn.Conv2d(dim+8,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.ds_2=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_2=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.ds_3=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_3=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.projector=nn.Conv2d(in_channels=4*dim,out_channels=self.symbol_dim,kernel_size=1, groups=4)

        self.channel=AWGNChannel(SNRdB)
        
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=4*dim,kernel_size=1, groups=4)
        self.dec_3=nn.Sequential(nn.Conv2d(2*dim,dim,1,groups=dim),resNetBlock(dim),resNetBlock(dim))
        self.us_3=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_2=nn.Sequential(nn.Conv2d(2*dim,dim,1,groups=dim),resNetBlock(dim),resNetBlock(dim))
        self.us_2=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_1=nn.Sequential(nn.Conv2d(2*dim,dim,1,groups=dim),resNetBlock(dim),resNetBlock(dim))
        self.us_1=nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr
            
        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)
        
        x1=self.ds_1(x)
        _,_,H1,W1=x1.shape
        x2=self.enc_1(torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1))
        x2=self.ds_2(x2)
        _,_,H2,W2=x2.shape
        x3=self.enc_2(x2)
        x3=self.ds_3(x3)
        _,_,H3,W3=x3.shape
        x4=self.enc_3(x3)

        # Resolution sync
        x1=torch.nn.functional.adaptive_avg_pool2d(x1, (H3,W3))
        x2=torch.nn.functional.adaptive_avg_pool2d(x2, (H3,W3))
        x3=torch.nn.functional.adaptive_avg_pool2d(x3, (H3,W3))
        total=torch.cat((x4,x3,x2,x1),dim=1)

        # Sending
        latent=self.projector(total)
        latent=latent[:,:self.chunk_size*chunk_num]
        y=rearrange(latent, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)
        y=torch.complex(y[...,0],y[...,1])
        power=torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)

        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H3, w=W3)

        # decoding
        noisy_latent=torch.zeros((B,self.symbol_dim,H3,W3),device=x.device)
        noisy_latent[:,:y_hat.shape[1]]=y_hat
        noisy_latent=self.deprojector(torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1))
        y4,y3,y2,y1=torch.split(noisy_latent,self.dim, dim=1)
        
        y3=torch.nn.functional.interpolate(y3,(H3,W3),mode='bilinear')
        y2=torch.nn.functional.interpolate(y2,(H2,W2),mode='bilinear')
        y1=torch.nn.functional.interpolate(y1,(H1,W1),mode='bilinear')
        
        temp=torch.zeros(B,2*self.dim, H3, W3, device=x.device)
        temp[:, 0::2]=y4
        temp[:, 1::2]=y3
        y3=self.dec_3(temp)
        y3=self.us_3(y3)
        
        temp=torch.zeros(B,2*self.dim, H2, W2, device=x.device)
        temp[:, 0::2]=y3
        temp[:, 1::2]=y2
        y2=self.dec_2(temp)
        y2=self.us_2(y2)
        
        temp=torch.zeros(B,2*self.dim, H1, W1, device=x.device)
        temp[:, 0::2]=y2
        temp[:, 1::2]=y1
        y1=self.dec_1(temp)
        
        x_hat=self.us_1(y1)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat

class ResNet_multi2(nn.Module):
    def __init__(self, dim, spp, SNRdB):
        super().__init__()
        self.ds_factor=2**4
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(spp*(self.ds_factor**2)*2)
        self.symbol_dim=round(rwa_symbol_dim/4)*4
        self.spp=self.symbol_dim/(2*self.ds_factor**2)
        self.F=4 # max number of chunks, spp approximately 0.5
        self.chunk_size=round(rwa_symbol_dim/4)
        self.min_F=1 # minimum number of chunks, spp approximately 0.1

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_1=nn.Sequential(nn.Conv2d(dim+8,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.ds_2=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_2=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.ds_3=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_3=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.projector=nn.Conv2d(in_channels=4*dim,out_channels=self.symbol_dim,kernel_size=1, groups=4)

        self.channel=AWGNChannel(SNRdB)
        
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=4*dim,kernel_size=1, groups=4)
        self.dec_3=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_3=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_2=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_2=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_1=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_1=nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr
            
        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)
        
        x1=self.ds_1(x)
        _,_,H1,W1=x1.shape
        x2=self.enc_1(torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1))
        x2=self.ds_2(x2)
        x1=self.ds_2(x1) # resolution sync
        _,_,H2,W2=x2.shape
        x3=self.enc_2(x2)
        x3=self.ds_3(x3)
        x2=self.ds_3(x2) # resolution sync
        x1=self.ds_3(x1) # resolution sync
        _,_,H3,W3=x3.shape
        x4=self.enc_3(x3)

        # Resolution sync
        total=torch.cat((x4,x3,x2,x1),dim=1)

        # Sending
        latent=self.projector(total)
        latent=latent[:,:self.chunk_size*chunk_num]
        y=rearrange(latent, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)
        y=torch.complex(y[...,0],y[...,1])
        power=torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)

        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H3, w=W3)

        # decoding
        noisy_latent=torch.zeros((B,self.symbol_dim,H3,W3),device=x.device)
        noisy_latent[:,:y_hat.shape[1]]=y_hat
        noisy_latent=self.deprojector(torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1))
        y4,y3,y2,y1=torch.split(noisy_latent,self.dim, dim=1)
        
        y3=self.dec_3(torch.cat((y4,y3),dim=1))
        y3=self.us_3(y3)
        y2=self.us_3(y2)  # resolution sync
        y1=self.us_3(y1)  # resolution sync
        
        y2=self.dec_2(torch.cat((y3,y2), dim=1))
        y2=self.us_2(y2)
        y1=self.us_2(y1)  # resolution sync
        
        y1=self.dec_1(torch.cat((y2,y1),dim=1))
        x_hat=self.us_1(y1)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat
    
class ResNet_multi3(nn.Module):
    def __init__(self, dim, spp, SNRdB):
        super().__init__()
        self.ds_factor=2**4
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(spp*(self.ds_factor**2)*2)
        self.symbol_dim=round(rwa_symbol_dim/4)*4
        self.spp=self.symbol_dim/(2*self.ds_factor**2)
        self.F=4 # max number of chunks, spp approximately 0.5
        self.chunk_size=round(rwa_symbol_dim/4)
        self.min_F=1 # minimum number of chunks, spp approximately 0.1

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_1=nn.Sequential(nn.Conv2d(dim+8,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.ds_2=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_2=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.ds_3=nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1)
        self.enc_3=nn.Sequential(resNetBlock(dim),resNetBlock(dim))
        self.sync1=nn.Conv2d(dim,dim,4,4)
        self.sync2=nn.Conv2d(dim,dim,2,2)
        self.projector=nn.Conv2d(in_channels=4*dim,out_channels=self.symbol_dim,kernel_size=1, groups=4)

        self.channel=AWGNChannel(SNRdB)
        
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=4*dim,kernel_size=1, groups=4)
        self.usync2=nn.ConvTranspose2d(dim,dim,2,2)
        self.usync1=nn.ConvTranspose2d(dim,dim,4,4)
        self.dec_3=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_3=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_2=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_2=nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_1=nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim))
        self.us_1=nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)

    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr
            
        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)
        
        x1=self.ds_1(x)
        _,_,H1,W1=x1.shape
        x2=self.enc_1(torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1))
        x2=self.ds_2(x2)
        _,_,H2,W2=x2.shape
        x3=self.enc_2(x2)
        x3=self.ds_3(x3)
        _,_,H3,W3=x3.shape
        x4=self.enc_3(x3)

        # Resolution sync
        x1=self.sync1(x1)
        x2=self.sync2(x2)
        total=torch.cat((x4,x3,x2,x1),dim=1)

        # Sending
        latent=self.projector(total)
        latent=latent[:,:self.chunk_size*chunk_num]
        y=rearrange(latent, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)
        y=torch.complex(y[...,0],y[...,1])
        power=torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)

        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        y_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H3, w=W3)

        # decoding
        noisy_latent=torch.zeros((B,self.symbol_dim,H3,W3),device=x.device)
        noisy_latent[:,:y_hat.shape[1]]=y_hat
        noisy_latent=self.deprojector(torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1))
        y4,y3,y2,y1=torch.split(noisy_latent,self.dim, dim=1)
        
        y2=self.usync2(y2)
        y1=self.usync1(y1)
        y3=self.dec_3(torch.cat((y4,y3),dim=1))
        y3=self.us_3(y3)
        y2=self.dec_2(torch.cat((y3,y2), dim=1))
        y2=self.us_2(y2)
        y1=self.dec_1(torch.cat((y2,y1),dim=1))
        x_hat=self.us_1(y1)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat

from torchinfo import summary
print("model baseline")
print(summary(ResNet_baseline(96,0.5,10), (1,3,256,256), depth=0))
print("model 1")
print(summary(ResNet_multi1(96,0.5,10), (1,3,256,256), depth=0))
print("model 1-group")
print(summary(ResNet_multi1group(96,0.5,10), (1,3,256,256), depth=0))
print("model 2")
print(summary(ResNet_multi2(96,0.5,10), (1,3,256,256), depth=0))
print("model 3")
print(summary(ResNet_multi3(96,0.5,10), (1,3,256,256), depth=0))