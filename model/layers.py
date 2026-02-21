import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from compressai.layers import conv3x3, conv1x1
from timm.layers import trunc_normal_, DropPath, to_2tuple
from einops import rearrange

__all__ = [
    "AWGNChannel",
    "resNetBlock",
    "PatchEmbedding", "PatchReconstruct", "PatchMerging", "PatchReverse", #"SWAttBlock",
    "NeXtBlock", #"LayerNorm"
]

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
    
class ChannelLastGroupedLinear(nn.Module):
    """
    nn.Linear의 Grouped 버전 (BHWC 레이아웃 전용)
    nn.Conv2d(..., kernel_size=1, groups=g)와 수학적으로 동일하지만,
    Permute 없이 BHWC 상태에서 바로 연산합니다.
    """
    def __init__(self, in_features, out_features, groups=1, bias=True):
        super().__init__()
        assert in_features % groups == 0, "Input dimensions must be divisible by groups."
        assert out_features % groups == 0, "Output dimensions must be divisible by groups."
        
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        
        # 가중치 형상: (Groups, In_per_group, Out_per_group)
        self.weight = nn.Parameter(torch.Tensor(groups, in_features // groups, out_features // groups))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Init (Grouped Linear에 맞게 조정)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (..., in_features)
        # 1. View logic to separate groups
        # (..., groups, in_per_group)
        x_shape = x.shape
        x = x.view(*x_shape[:-1], self.groups, -1)
        
        # 2. Einsum을 사용해 그룹별 행렬곱 수행 (가장 효율적)
        # g: groups, i: in_per_group, o: out_per_group
        x = torch.einsum('...gi, gio -> ...go', x, self.weight)
        
        # 3. Flatten back
        x = x.reshape(*x_shape[:-1], self.out_features)
        
        if self.bias is not None:
            x = x + self.bias
        return x
    
class resNetBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = nn.Linear(in_ch,in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Linear(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x # (N, H, W, C)

        out = self.conv1(x)
        out = self.relu(out)
        out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        out = self.conv2(out)
        out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out
    
#%% Swin
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        # Use Conv2d for the initial embedding as it's highly optimized for stride operations
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, C, H, W -> B, H, W, C
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.ln(x)
        return x

class PatchReconstruct(nn.Module):
    def __init__(self, dim, patch_size=4):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.up = nn.Linear(dim, patch_size[0] * patch_size[1] * 3)
        self.ps_size = patch_size

    def forward(self, x):
        # x: B, H, W, C
        x = self.up(x)
        # PixelShuffle logic via rearrangement
        x = rearrange(x, 'b h w (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=self.ps_size[0], p2=self.ps_size[1], c=3)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.ln = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x):
        # x: B, H, W, C
        B, H, W, C = x.shape
        # Efficient merging without slicing copies
        x = x.view(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).reshape(B, H // 2, W // 2, 4 * C)
        x = self.ln(x)
        x = self.reduction(x)
        return x

class PatchReverse(nn.Module):
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim // 2
        self.expand = nn.Linear(dim, 2 * 2 * out_dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: B, H, W, C
        x = self.ln(x)
        x = self.expand(x)
        # Reverse merging: B, H, W, (2*2*C) -> B, 2H, 2W, C
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2)
        return x
'''
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.register_buffer("relative_position_index", self._get_rel_pos_index(window_size))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _get_rel_pos_index(self, window_size):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        return relative_coords.sum(-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SWAttBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.ln1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        # Pre-calculate attention mask for shifted window
        if self.shift_size > 0:
            img_mask = torch.zeros((1, window_size + 2, window_size + 2, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = rearrange(img_mask, 'b (h p1) (w p2) c -> (b h w) (p1 p2)', p1=window_size, p2=window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        # Expects x: [B, H, W, C]
        H, W = x.shape[1], x.shape[2]
        shortcut = x
        x = self.ln1(x)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = rearrange(shifted_x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        shifted_x = rearrange(attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', h=H//self.window_size, w=W//self.window_size, p1=self.window_size)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
        '''
    
#%% ConvNext
'''class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x'''
    
class NeXtBlock(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, window_size=7,drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x
    
#%% My New Block
