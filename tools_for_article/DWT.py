import torch
import torch.nn.functional as F

# ---------- 工具：确保高宽为偶数（stride=2 的 2x2 卷积不会越界） ----------
def _pad_even_hw(x_nchw, mode="reflect"):
    # x: (N, C, H, W)
    N, C, H, W = x_nchw.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x_nchw = F.pad(x_nchw, (0, pad_w, 0, pad_h), mode=mode)  # (left,right,top,bottom)
    return x_nchw, pad_h, pad_w

# ---------- Haar 小波的分析(分解)与合成(重建)滤波器 ----------
def _haar_filters(device=None, dtype=torch.float32):
    # 1D Haar
    inv_sqrt2 = 1.0 / (2.0**0.5)
    l = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=dtype, device=device)   # 低通
    h = torch.tensor([-inv_sqrt2, inv_sqrt2], dtype=dtype, device=device)  # 高通
    # 分离卷积 -> 2D 核
    LL = torch.outer(l, l)  # (2,2)
    LH = torch.outer(l, h)
    HL = torch.outer(h, l)
    HH = torch.outer(h, h)
    # 返回顺序约定：LL, LH(垂直细节), HL(水平细节), HH(对角细节)
    k = torch.stack([LL, LH, HL, HH], dim=0)  # (4, 2, 2)
    return k

def _make_analysis_weight(C, device, dtype):
    # groups=C 的 depthwise 卷积: (out_channels, 1, 2, 2)
    k2d = _haar_filters(device, dtype)  # (4,2,2)
    W = torch.zeros((4*C, 1, 2, 2), dtype=dtype, device=device)
    for c in range(C):
        W[4*c:4*c+4, 0, :, :] = k2d
    return W

def _make_synthesis_weight(C, device, dtype):
    # 对于正交的 Haar，小波逆变换可用转置卷积，权重与分析核相同即可
    return _make_analysis_weight(C, device, dtype)

# ---------- 2D DWT：支持 (N, H, W, C) 或 (N, C, H, W) ----------
@torch.no_grad()
def dwt2_haar(x, channels_last=True, pad_mode="reflect", return_tuple=True):
    """
    x: (N, H, W, C) 若 channels_last=True；否则 (N, C, H, W)
    返回：
      - 若 return_tuple=True： (LL, LH, HL, HH)，每个张量形状与通道相同，尺寸为 H/2 x W/2
      - 若 return_tuple=False： 将四个子带在通道维拼接，形状为 (N, H/2, W/2, 4C) 或 (N, 4C, H/2, W/2)
    """
    if channels_last:
        x = x.permute(0, 3, 1, 2).contiguous()  # -> NCHW

    N, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    x, pad_h, pad_w = _pad_even_hw(x, mode=pad_mode)
    W_a = _make_analysis_weight(C, device, dtype)

    # depthwise conv: groups=C
    y = F.conv2d(x, W_a, bias=None, stride=2, padding=0, groups=C)  # (N, 4C, H/2, W/2)

    # 拆分四个子带
    y = y.view(N, C, 4, y.shape[-2], y.shape[-1])  # (N, C, 4, H/2, W/2)
    LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]  # (N, C, H/2, W/2)

    if channels_last:
        def to_nhwc(t): return t.permute(0, 2, 3, 1).contiguous()
        if return_tuple:
            return to_nhwc(LL), to_nhwc(LH), to_nhwc(HL), to_nhwc(HH)
        else:
            y_cat = torch.cat([LL, LH, HL, HH], dim=1)  # (N, 4C, H/2, W/2)
            return to_nhwc(y_cat)
    else:
        if return_tuple:
            return LL, LH, HL, HH
        else:
            return torch.cat([LL, LH, HL, HH], dim=1)  # (N, 4C, H/2, W/2)

# ---------- 2D IDWT（重建） ----------
@torch.no_grad()
def idwt2_haar(LL, LH=None, HL=None, HH=None, channels_last=True, original_hw=None):
    """
    两种输入形式：
      1) 四元组：LL, LH, HL, HH（每个是子带，形状一致）
      2) 单张量：把 4 个子带在通道维拼接（最后一个参数 LH 置为 None 即可，函数会按 4 子带拆分）
    original_hw: 若正变换时为奇数尺寸，这里用于裁剪回原始 (H, W)
    返回：重建后的张量，形状与原始一致 (channels_last 与输入一致)
    """
    # 统一到 NCHW
    if LH is None and HL is None and HH is None:
        # 输入是拼接后的张量
        y = LL
        if channels_last:
            y = y.permute(0, 3, 1, 2).contiguous()  # (N, 4C, H/2, W/2)
        N, C4, H2, W2 = y.shape
        assert C4 % 4 == 0, "通道数必须是 4 的倍数（包含四个子带）"
        C = C4 // 4
        y = y.view(N, C, 4, H2, W2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
    else:
        # 四元组输入
        def to_nchw(t):
            if channels_last:
                return t.permute(0, 3, 1, 2).contiguous()
            return t
        LL, LH, HL, HH = map(to_nchw, (LL, LH, HL, HH))
        N, C, H2, W2 = LL.shape

    device, dtype = LL.device, LL.dtype
    W_s = _make_synthesis_weight(C, device, dtype)

    # 拼接回 (N, 4C, H/2, W/2)
    y = torch.stack([LL, LH, HL, HH], dim=2).reshape(N, 4*C, H2, W2)
    x_rec = F.conv_transpose2d(y, W_s, bias=None, stride=2, padding=0, groups=C)  # (N, C, H, W)

    # 如果原始尺寸是奇数，这里裁剪回去
    if original_hw is not None:
        H_orig, W_orig = original_hw
        x_rec = x_rec[:, :, :H_orig, :W_orig]

    if channels_last:
        return x_rec.permute(0, 2, 3, 1).contiguous()
    else:
        return x_rec

# ---------- 多层分解/重建 ----------
@torch.no_grad()
def dwt2_haar_multilevel(x, levels=1, channels_last=True):
    """
    递归做多层分解。
    返回：pyramid dict，包含：
      - 'approx': 最后一层的 LL
      - 'details': 列表，从第1层到第L层的 (LH, HL, HH)
      - 'hw': 原始 (H, W)
    """
    if channels_last:
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
    else:
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
    pyr = {'details': [], 'hw': (H, W)}

    cur = x
    for _ in range(levels):
        LL, LH, HL, HH = dwt2_haar(cur, channels_last=channels_last, return_tuple=True)
        pyr['details'].append((LH, HL, HH))
        cur = LL
    pyr['approx'] = cur
    return pyr

@torch.no_grad()
def idwt2_haar_multilevel(pyr, channels_last=True):
    """
    根据 dwt2_haar_multilevel 的输出重建。
    """
    x = pyr['approx']
    # 注意：重建要从最深层往回
    for (LH, HL, HH) in reversed(pyr['details']):
        x = idwt2_haar(x, LH, HL, HH, channels_last=channels_last)
    # 裁剪到原始尺寸（如果有必要）
    return x[..., :pyr['hw'][0], :pyr['hw'][1], :] if channels_last else x[:, :, :pyr['hw'][0], :pyr['hw'][1]]


if __name__ == '__main__':
    x = torch.randn(8, 256, 256, 3)  # (N,H,W,C) channels-last

    # 1) 做一层 2D 小波分解
    LL, LH, HL, HH = dwt2_haar(x, channels_last=True, return_tuple=True)
    # 或者：y = dwt2_haar(x, channels_last=True, return_tuple=False)  # 通道拼接成 4C

    # 2) 用四个子带重建
    x_rec = idwt2_haar(LL, LH, HL, HH, channels_last=True, original_hw=(x.shape[1], x.shape[2]))
    print((x - x_rec).abs().max())  # Haar+float32 下误差应接近 1e-6~1e-7

    # 3) 多层分解与重建
    pyr = dwt2_haar_multilevel(x, levels=3, channels_last=True)
    x_rec2 = idwt2_haar_multilevel(pyr, channels_last=True)
    print((x - x_rec2).abs().max())
