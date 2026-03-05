import torch
import torch.nn as nn
from einops import rearrange
from model.layers import resNetBlock, AWGNChannel
import math

DEPTH=4
LAYERS=[2,2,6,2]
MIN_SPP=0.25
MAX_SPP=0.5
SPP_STEP=0.05

class ResNet_baseline(nn.Module):
    '''
    전통적인 방식대로 downsampling할 때마다 feature dimension을 2배씩 늘림.
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        encoder=[nn.Conv2d(dim+8,dim,1),]
        decoder=[nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1),]
        for i in range(self.depth):
            for _ in range(LAYERS[i]):
                encoder.append(resNetBlock(dim*2**i)) 
                decoder.insert(0, resNetBlock(dim*2**i))
            encoder.append(nn.Conv2d(in_channels=dim*2**i,
                                     out_channels=dim*2**(i+1),
                                     kernel_size=3,stride=2,padding=1) if i!=self.depth-1 else 
                                     nn.Conv2d(in_channels=dim*2**i,out_channels=self.symbol_dim,kernel_size=1))
            decoder.insert(0, nn.ConvTranspose2d(in_channels=dim*2**(i+1),
                                                 out_channels=dim*2**i,
                                                 kernel_size=3,
                                                 stride=2,padding=1,output_padding=1) if i!=self.depth-1 else
                                                 nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*2**(self.depth-1),kernel_size=1))
        self.encoder=nn.Sequential(*encoder)

        self.channel=AWGNChannel(SNRdB)

        self.decoder=nn.Sequential( *decoder)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
        
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
        x_cat=torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1)
        latent=self.encoder(x_cat)
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
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat

class ResNet_baseline_group(nn.Module):
    '''
    전통적인 방식대로 dimension은 늘렸지만 resNetBlock에 grouped Convolution을 주력으로 썼음.\n
    이러면 주요 병목은 downsampling layer & upsampling Layer가 됨.
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        encoder=[nn.Conv2d(dim+8,dim,1),]
        decoder=[nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1),]
        for i in range(self.depth):
            for _ in range(LAYERS[i]):
                encoder.append(resNetBlock(dim*2**i, groups=2**i)) 
                decoder.insert(0, resNetBlock(dim*2**i, groups=2**i))
            encoder.append(nn.Conv2d(in_channels=dim*2**i,
                                     out_channels=dim*2**(i+1),
                                     kernel_size=3,stride=2,padding=1, groups=2**i) if i!=self.depth-1 else 
                                     nn.Conv2d(in_channels=dim*2**i,out_channels=self.symbol_dim,kernel_size=1))
            decoder.insert(0, nn.ConvTranspose2d(in_channels=dim*2**(i+1),
                                                 out_channels=dim*2**i,
                                                 kernel_size=3,
                                                 stride=2,padding=1,output_padding=1, groups=2**i) if i!=self.depth-1 else
                                                 nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*2**(self.depth-1),kernel_size=1))
        self.encoder=nn.Sequential(*encoder)

        self.channel=AWGNChannel(SNRdB)

        self.decoder=nn.Sequential( *decoder)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
        
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
        x_cat=torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1)
        latent=self.encoder(x_cat)
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
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat
    
class ResNet_single(nn.Module):
    '''
    전통적인 방식대로 downsampling할 때마다 feature dimension을 2배씩 늘림.
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.ds_1=nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)
        encoder=[nn.Conv2d(dim+8,dim,1),]
        decoder=[nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1),]
        for i in range(self.depth):
            for _ in range(LAYERS[i]):
                encoder.append(resNetBlock(dim)) 
                decoder.insert(0, resNetBlock(dim))
            encoder.append(nn.Conv2d(in_channels=dim,
                                     out_channels=dim,
                                     kernel_size=3,stride=2,padding=1) if i!=self.depth-1 else 
                                     nn.Conv2d(in_channels=dim,out_channels=self.symbol_dim,kernel_size=1))
            decoder.insert(0, nn.ConvTranspose2d(in_channels=dim,
                                                 out_channels=dim,
                                                 kernel_size=3,
                                                 stride=2,padding=1,output_padding=1) if i!=self.depth-1 else
                                                 nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim,kernel_size=1))
        self.encoder=nn.Sequential(*encoder)

        self.channel=AWGNChannel(SNRdB)

        self.decoder=nn.Sequential( *decoder)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
        
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
        x_cat=torch.cat((x1,sideinfo.expand(-1, -1, H1, W1)), dim=1)
        latent=self.encoder(x_cat)
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
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat

class ResNet_multi1(nn.Module):
    '''
    아키텍쳐 1.\n
    downsampling시 Dimension을 늘리지 않고 다만 branch를 만듦. Sync는 interpolate와 averagepooling으로
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.enc=nn.ModuleList()
        self.ds=nn.ModuleList([nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)])
        self.embedder=nn.Conv2d(dim+8,dim,1)
        self.dec=nn.ModuleList()
        self.us=nn.ModuleList()
        for _ in range(self.depth):
            self.enc.append(nn.Sequential(resNetBlock(dim),resNetBlock(dim)))
            self.ds.append(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1))
            self.dec.append(nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim)))
            self.us.append(nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1))
        self.ds=self.ds[:-1]
        self.us=self.us[:-1].append(nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1))

        self.projector=nn.Conv2d(in_channels=dim*self.depth,out_channels=self.symbol_dim,kernel_size=1)
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*self.depth,kernel_size=1)

        self.channel=AWGNChannel(SNRdB)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
    
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 1, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)

        xs=[]
        resolutions=[]
        temp=x
        for i in range(self.depth):
            temp=self.ds[i](temp)
            resolutions.append(temp.shape[2:])
            if i==0:
                temp=self.embedder(torch.cat((temp, sideinfo.expand(-1,-1,*temp.shape[2:])), dim=1))
            temp=self.enc[i](temp)
            xs.append(temp)
        # Resolution sync
        H3, W3=resolutions[-1]
        for i in range(len(xs)):
            xs[i]=torch.nn.functional.adaptive_avg_pool2d(xs[i], (H3,W3))
        total=torch.cat(xs, dim=1)

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
        temp=torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1)
        noisy_latent=self.deprojector(temp)
        ys=list(torch.split(noisy_latent,self.dim, dim=1))

        ys.append(torch.zeros_like(ys[-1]))
        for i in range(self.depth):
            temp=self.dec[i](torch.cat((
                torch.nn.functional.interpolate(ys[-1], resolutions[-(i+1)], mode='bilinear'),
                torch.nn.functional.interpolate(ys[-2], resolutions[-(i+1)], mode='bilinear')
                 ),dim=1))
            ys=ys[:-2]
            ys.append(self.us[i](temp))
        x_hat=torch.clamp(ys[0], 0, 1)
        
        return x_hat

class ResNet_multi1group(nn.Module):
    '''
    아키텍쳐 1.\n
    downsampling시 Dimension을 늘리지 않고 다만 branch를 만듦. Sync는 interpolate와 averagepooling으로\n
    그리고 얘는 merging 시 group을 씀
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.enc=nn.ModuleList()
        self.ds=nn.ModuleList([nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)])
        self.embedder=nn.Conv2d(dim+8,dim,1)
        self.dec=nn.ModuleList()
        self.us=nn.ModuleList()
        for _ in range(self.depth):
            self.enc.append(nn.Sequential(resNetBlock(dim),resNetBlock(dim)))
            self.ds.append(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1))
            self.dec.append(nn.Sequential(nn.Conv2d(2*dim,dim,1, groups=dim),resNetBlock(dim),resNetBlock(dim)))
            self.us.append(nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1))
        self.ds=self.ds[:-1]
        self.us=self.us[:-1].append(nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1))

        self.projector=nn.Conv2d(in_channels=dim*self.depth,out_channels=self.symbol_dim,kernel_size=1)
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*self.depth,kernel_size=1)

        self.channel=AWGNChannel(SNRdB)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
    
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 1, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)

        xs=[]
        resolutions=[]
        temp=x
        for i in range(self.depth):
            temp=self.ds[i](temp)
            resolutions.append(temp.shape[2:])
            if i==0:
                temp=self.embedder(torch.cat((temp, sideinfo.expand(-1,-1,*temp.shape[2:])), dim=1))
            temp=self.enc[i](temp)
            xs.append(temp)
        # Resolution sync
        H3, W3=resolutions[-1]
        for i in range(len(xs)):
            xs[i]=torch.nn.functional.adaptive_avg_pool2d(xs[i], (H3,W3))
        total=torch.cat(xs, dim=1)

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
        temp=torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1)
        noisy_latent=self.deprojector(temp)
        ys=list(torch.split(noisy_latent,self.dim, dim=1))

        ys.append(torch.zeros_like(ys[-1]))
        for i in range(self.depth):
            temp=self.dec[i](rearrange(
                torch.stack((
                torch.nn.functional.interpolate(ys[-1], resolutions[-(i+1)], mode='bilinear'),
                torch.nn.functional.interpolate(ys[-2], resolutions[-(i+1)], mode='bilinear')
                 ),dim=1), 'b t c h w -> b (t c) h w'))
            ys=ys[:-2]
            ys.append(self.us[i](temp))
        x_hat=torch.clamp(ys[0], 0, 1)
        
        return x_hat

class ResNet_multi2(nn.Module):
    '''
    아키텍쳐 2.\n
    downsampling시 Dimension을 늘리지 않고 다만 branch를 만듦. Sync는 ds와 us 전부 통과.
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.enc=nn.ModuleList()
        self.ds=nn.ModuleList([nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)])
        self.embedder=nn.Conv2d(dim+8,dim,1)
        self.dec=nn.ModuleList()
        self.us=nn.ModuleList()
        for _ in range(self.depth):
            self.enc.append(nn.Sequential(resNetBlock(dim),resNetBlock(dim)))
            self.ds.append(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1))
            self.dec.append(nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim)))
            self.us.append(nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1))
        self.ds=self.ds[:-1]
        self.us=self.us[:-1].append(nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1))

        self.projector=nn.Conv2d(in_channels=dim*self.depth,out_channels=self.symbol_dim,kernel_size=1)
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*self.depth,kernel_size=1)

        self.channel=AWGNChannel(SNRdB)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
    
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 1, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)

        xs=[x]
        for i in range(self.depth):
            for j in range(len(xs)):
                xs[j]=self.ds[i](xs[j])
            temp=xs[-1]
            if i==0:
                temp=self.embedder(torch.cat((temp, sideinfo.expand(-1,-1,*temp.shape[2:])), dim=1))
            temp=self.enc[i](temp)
            xs.append(temp)
        # Resolution sync
        total=torch.cat(xs[1:], dim=1)
        H3, W3=total.shape[2:]

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
        temp=torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1)
        noisy_latent=self.deprojector(temp)
        ys=list(torch.split(noisy_latent,self.dim, dim=1))

        ys.append(torch.zeros_like(ys[-1]))
        for i in range(self.depth):
            temp=self.dec[i](torch.cat((ys[-1],ys[-2]),dim=1))
            ys=ys[:-2]
            ys.append(temp)
            for j in range(len(ys)):
                ys[j]=self.us[i](ys[j])
        x_hat=torch.clamp(ys[0], 0, 1)
        
        return x_hat
    
class ResNet_multi3(nn.Module):
    '''
    아키텍쳐 2.\n
    downsampling시 Dimension을 늘리지 않고 다만 branch를 만듦. Sync는 Conv로 대체.
    '''
    def __init__(self, dim, spp=None, SNRdB:int=10, depth:int=4, **kwargs):
        super().__init__()
        self.spp=spp if spp is not None else MAX_SPP
        self.depth=depth if depth is not None else DEPTH
        self.ds_factor=2**self.depth
        self.snr=SNRdB
        self.dim=dim
        rwa_symbol_dim=int(self.spp*(self.ds_factor**2)*2)
        chunk_size=int(SPP_STEP*self.ds_factor**2*2)
        print(chunk_size)
        self.symbol_dim=math.ceil(rwa_symbol_dim/chunk_size)*chunk_size
        self.F=self.symbol_dim//chunk_size
        self.min_F=math.ceil(int(MIN_SPP*self.ds_factor**2*2)/chunk_size)
        self.chunk_size=chunk_size

        self.sideinfo_projector=nn.Linear(2,8)
        
        self.enc=nn.ModuleList()
        self.ds=nn.ModuleList([nn.Conv2d(in_channels=3,out_channels=dim,kernel_size=3,stride=2,padding=1)])
        self.embedder=nn.Conv2d(dim+8,dim,1)
        self.dec=nn.ModuleList()
        self.us=nn.ModuleList()
        for _ in range(self.depth):
            self.enc.append(nn.Sequential(resNetBlock(dim),resNetBlock(dim)))
            self.ds.append(nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1))
            self.dec.append(nn.Sequential(nn.Conv2d(2*dim,dim,1),resNetBlock(dim),resNetBlock(dim)))
            self.us.append(nn.ConvTranspose2d(in_channels=dim,out_channels=dim,kernel_size=3,stride=2,padding=1,output_padding=1))
        self.ds=self.ds[:-1]
        self.us=self.us[:-1].append(nn.ConvTranspose2d(in_channels=dim,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1))

        self.projector=nn.Conv2d(in_channels=dim*self.depth,out_channels=self.symbol_dim,kernel_size=1)
        self.deprojector=nn.Conv2d(in_channels=self.symbol_dim+8,out_channels=dim*self.depth,kernel_size=1)
        
        self.syncer=nn.ModuleList([nn.Conv2d(dim,dim,2**(self.depth-1-i),2**(self.depth-1-i)) for i in range(self.depth-1)])
        self.desyncer=nn.ModuleList([nn.Identity()]+[nn.ConvTranspose2d(dim,dim,2**i,2**i) for i in range(1,self.depth)])
        self.channel=AWGNChannel(SNRdB)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dim', type=int)
        parser.add_argument('--model.img_shape', type=tuple, default=(256,256))
        parser.add_argument('--model.depth', type=int, default=4)
        return parser
    
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None, chunk_num:int=4):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        chunk_info=torch.ones_like(snrs)*chunk_num  # (B,1)
        sideinfo:torch.Tensor=self.sideinfo_projector(torch.cat((snrs, chunk_info),dim=1))
        # Add spatial dimensions to sideinfo: (B, 8) -> (B, 1, 8, 1, 1)
        sideinfo = sideinfo.unsqueeze(-1).unsqueeze(-1)

        xs=[]
        resolutions=[]
        temp=x
        for i in range(self.depth):
            temp=self.ds[i](temp)
            resolutions.append(temp.shape[2:])
            if i==0:
                temp=self.embedder(torch.cat((temp, sideinfo.expand(-1,-1,*temp.shape[2:])), dim=1))
            temp=self.enc[i](temp)
            xs.append(temp)
        # Resolution sync
        H3, W3=resolutions[-1]
        for i in range(len(xs)-1):
            xs[i]=self.syncer[i](xs[i])
        total=torch.cat(xs, dim=1)

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
        temp=torch.cat((noisy_latent,sideinfo.expand(-1,-1,H3,W3)),dim=1)
        noisy_latent=self.deprojector(temp)
        ys=list(torch.split(noisy_latent,self.dim, dim=1))

        ys.append(torch.zeros_like(ys[-1]))
        for i in range(self.depth):
            temp=self.dec[i](torch.cat((
                ys[-1],
                self.desyncer[i](ys[-2])
                ),dim=1))
            ys=ys[:-2]
            ys.append(self.us[i](temp))
        x_hat=torch.clamp(ys[0], 0, 1)
        
        return x_hat