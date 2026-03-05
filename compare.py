import os
import random
import yaml
import torch

import torch.nn as nn
import torch
import model as jscc_models
from model.layers import AWGNChannel
from utils import load_dataloaders, getUsableGPUs
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
     ('methodcomp/baseline', 'Baseline'),
     ('methodcomp/baseline_group', 'Baseline(grouped)'),
     ('methodcomp/single', 'Single'),
     ('methodcomp/multi1', 'Multi(interpolate)'),
     ('methodcomp/multi1group', 'Grouped Multi(interpolate)'),
     ('methodcomp/multi2', 'Multi(shared ds)'),
     ('methodcomp/multi3', 'Multi(seperate ds)'),
#    ('BASELINE/baseline', 'DeepJSCCl++ baseline'),
#    ('BASELINE/ConvNext', 'ConvNext baseline'),
#    ('BASELINE/RESNET', 'ResNet baseline'),
#    ('SINGLE/baseline', 'DeepJSCCl++ single'),
#    ('SINGLE/ConvNext', 'ConvNext single'),
#    ('SINGLE/RESNET', 'ResNet single'),
#    ('MULTIEXTRACT/baseline', 'DeepJSCCl++ multi'),
#    ('MULTIEXTRACT/ConvNext', 'ConvNext multi'),
#    ('MULTIEXTRACT/RESNET', 'ResNet multi'),
    ]

Params={
     'Baseline': round(21257361/1e6,2),
     'Baseline(grouped)': round(5350545/1e6,2),
     'Single': round(1266897/1e6,2),
     'Multi(interpolate)': round(1188849/1e6,2),
     'Grouped Multi(interpolate)': round(1115889/1e6,2),
     'Multi(shared ds)': round(1188849/1e6,2),
     'Multi(seperate ds)': round(2737713/1e6,2),
}
MACs={
     'Baseline': 22.44,
     'Baseline(grouped)': 11.08,
     'Single': 5.49,
     'Multi(interpolate)': 5.67,
     'Grouped Multi(interpolate)': 5.27,
     'Multi(shared ds)': 8.54,
     'Multi(seperate ds)': 16.17,
}

rate_psnrs={}
snr_psnrs={}

for logdir, name in test_list:
    save_dir=os.path.join('logs', logdir)
    with open(os.path.join(save_dir,'config.yaml')) as f:
        config=yaml.unsafe_load(f)
    print(config)

    gpu_id=1#getUsableGPUs()
    if gpu_id is not None:
        print("Using CUDA")
        torch.cuda.set_device(gpu_id)
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
    model:jscc_models.ResNet_baseline = model_class(**cfg_model).cuda()
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
    
    snrs=[1,4,7,10,13,16,19,22,25]
    spps=[i*model.chunk_size/(model.ds_factor**2*2) for i in range(model.min_F, model.F+1)]
    # loop for rate first
    total_images=len(train_dl.dataset)
    loss=nn.MSELoss()
    with torch.no_grad():
        print(model.F, model.min_F)
        loss_tracker=torch.zeros(model.F-model.min_F+1).cuda()
        print(loss_tracker.shape)
        loss_tracker1=torch.zeros(len(snrs)).cuda()
        print(loss_tracker1.shape)
        for data in tqdm(train_dl, desc='Training Dataset'):
            data:torch.Tensor=data.cuda()
            model.snr=10
            model.channel=AWGNChannel(10)
            for i, chunk_n in enumerate(range(model.min_F, model.F+1)):
                x_hat=model.forward(data, chunk_num=chunk_n)
                mseloss=loss(data, x_hat)
                loss_tracker[i]+=mseloss*data.shape[0]
            for i,snr in enumerate(snrs):
                model.snr=snr
                model.channel=AWGNChannel(snr)
                x_hat=model.forward(data)
                mseloss=loss(data, x_hat)
                loss_tracker1[i]+=mseloss*data.shape[0]
        rate_psnrs[name]['train']=loss_tracker/len(train_dl.dataset)
        snr_psnrs[name]['train']=loss_tracker1/len(train_dl.dataset)
        
        loss_tracker=torch.zeros(model.F-model.min_F+1).cuda()
        loss_tracker1=torch.zeros(len(snrs)).cuda()
        for data in tqdm(valid_dl, desc='Valid Dataset'):
            data:torch.Tensor=data.cuda()
            model.snr=10
            model.channel=AWGNChannel(10)
            for i, chunk_n in enumerate(range(model.min_F, model.F+1)):
                x_hat=model.forward(data, chunk_num=chunk_n)
                mseloss=loss(data, x_hat)
                loss_tracker[i]+=mseloss*data.shape[0]
            for i, snr in enumerate(snrs):
                model.snr=snr
                model.channel=AWGNChannel(snr)
                x_hat=model.forward(data)
                mseloss=loss(data, x_hat)
                loss_tracker1[i]+=mseloss*data.shape[0]
        rate_psnrs[name]['valid']=loss_tracker/len(valid_dl.dataset)
        snr_psnrs[name]['valid']=loss_tracker1/len(valid_dl.dataset)
        
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) JSON 저장용 유틸
# ----------------------------
def _to_builtin(x):
    """torch.Tensor / numpy / scalar / list 등을 JSON 직렬화 가능한 파이썬 타입으로 변환"""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.floating, np.integer)):
        return x.item()

    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]

    return x

def save_results_json(rate_psnrs, snr_psnrs, out_dir="results", filename="eval_psnr_results.json"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    payload = {
        "rate_psnrs": _to_builtin(rate_psnrs),
        "snr_psnrs": _to_builtin(snr_psnrs),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")
    return out_path

# ----------------------------
# 2) PSNR 변환 및 x축 구성 유틸
# ----------------------------
def mse_to_psnr(mse, pixel_max=1.0, eps=1e-12):
    """
    mse: array-like
    return: np.ndarray PSNR(dB)
    """
    mse = np.asarray(mse, dtype=np.float64)
    mse = np.maximum(mse, eps)
    return 10.0 * np.log10((pixel_max ** 2) / mse)

def _extract_series(d, split_prefer=("valid", "train")):
    """
    d: rate_psnrs[name] 또는 snr_psnrs[name] 같은 구조 기대
    return: (split_name, np.ndarray values) or (None, None)
    """
    if not isinstance(d, dict):
        return None, None
    for sp in split_prefer:
        if sp in d:
            return sp, np.asarray(_to_builtin(d[sp]), dtype=np.float64)
    # 혹시 dict가 바로 시퀀스(리스트)로 들어왔으면
    try:
        arr = np.asarray(_to_builtin(d), dtype=np.float64)
        return "unknown", arr
    except Exception:
        return None, None

# ----------------------------
# 3) 3x1 subplot 플로팅
# ----------------------------
def plot_all(Params, MACs, rate_psnrs, snr_psnrs,
             snrs=(1,4,7,10,13,16,19,22,25),
             MIN_SPP=0.25, MAX_SPP=0.5, SPP_STEP=0.05,
             pixel_max=1.0,
             save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.5), constrained_layout=True)

    # (1) Params vs MACs scatter
    ax = axes[0]
    for name in Params.keys():
        if name not in MACs:
            continue
        ax.scatter(Params[name], MACs[name], label=name)
        # 필요하면 점 옆에 텍스트 표시:
        # ax.annotate(name, (Params[name], MACs[name]), fontsize=8, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("Params (M)")
    ax.set_ylabel("MACs (G)")
    ax.set_title("Model Complexity")
    ax.grid(True, which="both")
    ax.legend(loc="best", ncols=2)

    # (2) rate_psnrs 비교: spp vs PSNR
    ax = axes[1]
    for model_name, d in rate_psnrs.items():
        split, mse_arr = _extract_series(d)
        if mse_arr is None or mse_arr.size == 0:
            continue
        L = min(len(spps), len(mse_arr))
        spp_x = np.array(spps[:L], dtype=np.float64)

        # mse_arr가 (F-min_F+1) 길이인 시퀀스라고 가정하고 spp 축 구성
        psnr_y = mse_to_psnr(mse_arr, pixel_max=pixel_max)

        ax.plot(spp_x, psnr_y, marker="o", label=model_name)

    ax.set_xlabel("spp")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate (spp) vs PSNR")
    ax.grid(True, which="both")
    ax.legend(loc="best", ncols=2)

    # (3) snr_psnrs 비교: snr vs PSNR
    ax = axes[2]
    snrs = list(snrs)
    for model_name, d in snr_psnrs.items():
        split, mse_arr = _extract_series(d)
        if mse_arr is None or mse_arr.size == 0:
            continue

        # 길이가 snrs랑 다르면 가능한 범위까지만 맞춰서 그림
        L = min(len(snrs), len(mse_arr))
        x = np.array(snrs[:L], dtype=np.float64)
        y = mse_to_psnr(mse_arr[:L], pixel_max=pixel_max)

        ax.plot(x, y, marker="o", label=model_name)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("SNR vs PSNR")
    ax.grid(True, which="both")
    ax.legend(loc="best", ncols=2)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[Saved figure] {save_path}")

    plt.show()
    return fig

# ----------------------------
# 4) 실행 예시 (당신 코드 끝부분에 추가)
# ----------------------------
# JSON 저장
json_path = save_results_json(rate_psnrs, snr_psnrs, out_dir="results", filename="eval_psnr_results.json")

# 플롯 저장/표시
plot_all(
    Params=Params,
    MACs=MACs,
    rate_psnrs=rate_psnrs,
    snr_psnrs=snr_psnrs,
    snrs=(1,4,7,10,13,16,19,22,25),
    MIN_SPP=0.25,
    MAX_SPP=0.5,
    SPP_STEP=0.05,
    pixel_max=1.0,                 # 데이터가 [0,1]이면 1.0 / [0,255]면 255.0
    save_path="results/plots_3x1.png"
)