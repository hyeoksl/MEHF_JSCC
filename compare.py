import os
import random
import yaml
import torch

import torch.nn as nn
import torch
import model as jscc_models
from model.layers import AWGNChannel
from utils import load_dataloaders, progressMeter, getUsableGPUs
from einops import rearrange
from tqdm import tqdm

# spp가 0.25인 상태로 SNR 1,4,7,10,13,16,19,22 테스트
# SNR이 10인 상태로 spp 0.1, 0.2, 0.3, 0.4, 0.5 테스트

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']  # 시스템에 있는 걸로 fallback
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5

test_list=[
#    ('BASELINE/baseline', 'DeepJSCCl++ baseline'),
#    ('BASELINE/ConvNext', 'ConvNext baseline'),
    ('BASELINE/RESNET', 'ResNet baseline'),
#    ('SINGLE/baseline', 'DeepJSCCl++ single'),
#    ('SINGLE/ConvNext', 'ConvNext single'),
    ('SINGLE/RESNET', 'ResNet single'),
#    ('MULTIEXTRACT/baseline', 'DeepJSCCl++ multi'),
#    ('MULTIEXTRACT/ConvNext', 'ConvNext multi'),
    ('MULTIEXTRACT/RESNET', 'ResNet multi'),
    ]

rate_psnrs={}
snr_psnrs={}

for logdir, name in test_list:
    save_dir=os.path.join('logs', logdir)
    with open(os.path.join(save_dir,'config.yaml')) as f:
        config=yaml.unsafe_load(f)
    print(config)

    gpu_id=getUsableGPUs()
    if gpu_id is not None:
        print("Using CUDA")
        torch.cuda.set_device(gpu_id)
        device=torch.device('cuda')
    else:
        print("No GPU Available!")
        break

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    cfg_dataset=config.dataset
    train_dl, valid_dl = load_dataloaders(**cfg_dataset)
    print(f"Number of dataset | train : {len(train_dl.dataset)}, valid : {len(valid_dl.dataset)}")
    
    # Model
    cfg_model = config.model
    print(cfg_model)
    model_class = getattr(jscc_models, cfg_model.type)
    del cfg_model.type
    model:jscc_models.ResNet = model_class(**cfg_model).cuda()
    H, W = cfg_model.img_shape
    
    # Load Checkpoint
    # ckpt에 best checkpoint 불러오는 거 하나.
    for item in os.listdir(save_dir):
        if 'best_ckpt' in item:
            ckpt = os.path.join(save_dir, item)
            break
    model.eval()
    checkpoint = torch.load(ckpt, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded from" + ckpt)
    
    rate_psnrs[name]={}
    snr_psnrs[name]={}
    
    # loop for rate first
    total_images=len(train_dl.dataset)
    model.snr=10
    model.channel=AWGNChannel(10)
    loss=nn.MSELoss()
    with torch.no_grad():
        for data in tqdm(train_dl, desc='Rate adaptive training'):
            data:torch.Tensor=data.cuda()
            y=model.encoder(data)
            y=y.permute(0,3,1,2) # BCHW
            B,C,H,W=y.shape
            zeros=torch.zeros_like(y)
            instsnr,_=model._get_inst_csi(B,data.device)
            for i in range(model.min_chunks, model.F+1):
                temp=y[:,:model.chunk_size*i]
                temp=rearrange(temp, 'b (n_c iq) h w -> b (n_c h w) iq',iq=2)
                temp_complex=torch.complex(temp[...,0],temp[...,1])
                
                power=torch.mean(torch.abs(temp_complex)**2,dim=1,keepdim=True)+1e-8
                temp_complex=temp_complex/torch.sqrt(power)

                temp_hat_complex=model.channel(temp_complex,instsnr)

                temp_hat=torch.stack((temp_hat_complex.real, temp_hat_complex.imag), dim=-1)
                temp_hat=rearrange(temp_hat, 'b (n_c h w) iq -> b (n_c iq) h w', h=H, w=W)
                temp_full=zeros
                temp_full[:,model.chunk_size*i]=temp_hat
                temp_full=temp_full.permute(0,2,3,1)
                x_hat=model.decoder(temp_full)
                x_hat=torch.clamp(x_hat, 0, 1)
                
                mseloss=loss(data, x_hat)
                rate_psnrs[name][]
            

            out_net=model(data)
            
    # loop for snrs last
