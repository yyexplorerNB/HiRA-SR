import cv2
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
import torch
from torch.nn.functional import conv2d

'''
一些代码修改自：
用于评估图像融合质量的函数https://github.com/sunbinuestc/VIF-metrics-analysis
并且对部分指标的计算偏差进行了统一。

格式：
可见光彩色图像image_A：形状为[H,W,BGR]，数据为uint8，numpy格式
红外线图像image_B：形状为[H,W]，数据为uint8，numpy格式
融合结果image_F：形状为[H,W,BGR]，数据为uint8，numpy格式
计算指标时会默认转换为灰度图计算

后缀为的为目前没有探明的计算误差原因
数据对比自《Probing Synergistic High-Order Interaction in Infrared and Visible Image Fusion》，以下称原文章
目前以探明的孙老师方法计算误差：
对于MI：孙老师使用e为底数，但原文章使用2为底数
对于VIF：孙老师计算结果与原文章相同。但是另一篇综述给出的matlab计算结果不同。
对于AG：孙老师在某个计算节点除以2.

评价指标含义：
SF:
空间频率，更高的SF，意味着更丰富的边缘和纹理细节
MI:
互信息，更高的MI，意味着源图像转移到融合图像中的信息越多
VIF:
视觉保真度，更高的VIF，意味着融合结果越符合人类视觉感知
Qabf:
边缘保持度，更高的Qabf，意味着有更多的边缘信息从源图像转移到融合图像
AG:
平均梯度，更高的AG，意味着更丰富的梯度信息和纹理细节。
EN:
信息熵，更高的EN，意味着更多的信息量
Qcb:
Chen-Blum 指标,越高越好，是Qcv的改进版。
Qcv:
Chen-Varshney 指标，越小越好，从人类视觉系统上衡量融合效果
'''
@METRIC_REGISTRY.register()
def calculate_SF(image_F,only_gray=True,need_normalize=True,**kwargs):
    """
    计算融合图像的空间频率 (Spatial Frequency, SF)。
    
    参数:
    - image_F: 融合后的彩色图像，形状为 [H, W, BGR]，uint8 格式。
    need_normalize: 是否将uint设置为0~1范围的float

    返回:
    - sf: 空间频率的值。
    """
    # 转换融合图像为灰度图
    gray_img = tool_adaptive_BGR2GRAY(image_F)
    assert only_gray # 
    if need_normalize:
        gray_img = gray_img/255.
    # 获取图像尺寸
    H, W = gray_img.shape

    # 计算行频率（RF）
    rf = np.sqrt(np.sum((gray_img[:, 1:] - gray_img[:, :-1]) ** 2) / (H * W))

    # 计算列频率（CF）
    cf = np.sqrt(np.sum((gray_img[1:, :] - gray_img[:-1, :]) ** 2) / (H * W))

    # 计算空间频率（SF）
    sf = np.sqrt(rf ** 2 + cf ** 2)

    return sf

@METRIC_REGISTRY.register()
def calculate_MI(image_A,image_B,image_F,resize=False,log_by_2=True,**kwargs):
    '''
    计算图像之间的互信息 (Mutual Information, MI)。
    需要注意的是：计算对数时应当使用2为底，而不是采用e。这是shanon熵定理规定的。
    '''
    def entropy(image):
        """
        使用 OpenCV 和 NumPy 计算图像的熵（Entropy）。
        
        参数:
        image : numpy.ndarray
            输入的灰度图像 (H, W)，需要是单通道灰度图像。
        log_by_2: bool
            计算熵时，采用2为底还是e为底
            
        返回:
        float
            图像的熵值。
        """
        # 计算图像的灰度直方图，256个灰度级
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # 将直方图归一化，得到每个灰度级的概率
        hist = hist / hist.sum()
        
        # 计算熵
        # 过滤掉为零的概率，防止计算 log(0)
        hist_nonzero = hist[hist > 0]
        if log_by_2:
            entropy_value = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        else:
            entropy_value = -np.sum(hist_nonzero * np.log(hist_nonzero))
        
        return entropy_value

    def Hab(grey_matrixA,grey_matrixB,grey_level=256):
        row, column= grey_matrixA.shape
        counter = np.zeros((grey_level, grey_level), dtype=int)
        grey_matrixA = grey_matrixA.astype(int) + 1  # Adjust indices to start from 1
        grey_matrixB = grey_matrixB.astype(int) + 1
        for i in range(row):
            for j in range(column):
                indexx = grey_matrixA[i, j]
                indexy = grey_matrixB[i, j]
                counter[indexx-1, indexy-1] += 1
        # Compute the joint probability distribution
        total = np.sum(counter)  # Total number of pixels
        nonzero_indices = counter != 0  # Indices where counter is not zero
        p = counter / total  # Joint probability distribution

        # Calculate the mutual information
        if log_by_2:
            return np.sum(-p[nonzero_indices] * np.log2(p[nonzero_indices]))
        else:
            return np.sum(-p[nonzero_indices] * np.log(p[nonzero_indices]))
    
    grey_vi = tool_adaptive_BGR2GRAY(image_A)
    grey_fused = tool_adaptive_BGR2GRAY(image_F)
    if resize:
        grey_vi = tool_adaptive_resize(grey_vi,image_F)
        grey_fused = tool_adaptive_resize(grey_fused,image_F)
    HA,HB,HF=entropy(grey_vi),entropy(image_B),entropy(grey_fused)
    HFA=Hab(grey_fused,grey_vi,256)
    HFB=Hab(grey_fused,image_B,256)
    MIFA=HA+HF-HFA
    MIFB=HB+HF-HFB
    # print('yuanshi',HF)
    return  MIFA+MIFB

@METRIC_REGISTRY.register()
def calculate_VIF(image_F, image_A, image_B,resize=False, **kwargs):
    def getvif(preds, target, sigma_n_sq=2.0):
        def _filter(win_size, sigma, dtype, device):
            # This code is inspired by
            # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/utils.py#L45
            # https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/functional/filters.py#L38
            # Both links do the same, but the second one is cleaner
            coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
            g = coords**2
            g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma**2)) 

            g /= torch.sum(g)
            return g
        preds = torch.from_numpy(preds).float()
        target = torch.from_numpy(target).float()
        dtype = preds.dtype
        device = preds.device

        preds = preds.unsqueeze(0).unsqueeze(0)  # Add channel dimension
        target = target.unsqueeze(0).unsqueeze(0)
        # Constant for numerical stability
        eps = torch.tensor(1e-10, dtype=dtype, device=device)

        sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)

        preds_vif, target_vif = torch.zeros(1, dtype=dtype, device=device), torch.zeros(
            1, dtype=dtype, device=device
        )
        for scale in range(4):
            n = 2.0 ** (4 - scale) + 1
            kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]

            if scale > 0:
                target = conv2d(target.float(), kernel)[:, :, ::2, ::2]
                preds = conv2d(preds.float(), kernel)[:, :, ::2, ::2]

            mu_target = conv2d(target, kernel)
            mu_preds = conv2d(preds, kernel)
            mu_target_sq = mu_target**2
            mu_preds_sq = mu_preds**2
            mu_target_preds = mu_target * mu_preds

            if scale == 0:
                target = target.byte()
                preds = preds.byte()
            sigma_target_sq = torch.clamp(
                conv2d((target**2).float(), kernel) - mu_target_sq, min=0.0
            )
            sigma_preds_sq = torch.clamp(
                conv2d((preds**2).float(), kernel) - mu_preds_sq, min=0.0
            )
            sigma_target_preds = conv2d((target * preds).float(), kernel) - mu_target_preds

            g = sigma_target_preds / (sigma_target_sq + eps)
            sigma_v_sq = sigma_preds_sq - g * sigma_target_preds

            mask = sigma_target_sq < eps
            g[mask] = 0
            sigma_v_sq[mask] = sigma_preds_sq[mask]
            sigma_target_sq[mask] = 0

            mask = sigma_preds_sq < eps
            g[mask] = 0
            sigma_v_sq[mask] = 0

            mask = g < 0
            sigma_v_sq[mask] = sigma_preds_sq[mask]
            g[mask] = 0
            sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)

            preds_vif_scale = torch.log10(
                1.0 + (g**2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)
            )
            preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
            target_vif = target_vif + torch.sum(
                torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3]
            )
        vif = preds_vif / target_vif
        if torch.isnan(vif):
            return 1.0
        else:
            return vif.item()

    image_F = tool_adaptive_BGR2GRAY(image_F)
    image_A = tool_adaptive_BGR2GRAY(image_A)
    image_B = tool_adaptive_BGR2GRAY(image_B)
    if resize:
        image_A = tool_adaptive_resize(image_A,image_F)
        image_B = tool_adaptive_resize(image_B,image_F)

    return getvif(image_F, image_A) + getvif(image_F, image_B)


    
@METRIC_REGISTRY.register()
def calculate_Qabf(image_F, image_A, image_B,resize=False, **kwargs):
    image_F = tool_adaptive_BGR2GRAY(image_F)
    image_A = tool_adaptive_BGR2GRAY(image_A)
    image_B = tool_adaptive_BGR2GRAY(image_B)
    if resize:
        image_A = tool_adaptive_resize(image_A,image_F)
        image_B = tool_adaptive_resize(image_B,image_F)
    def compute_gradients_and_orientations(image, h1, h3):
        gx = conv2d(image, h3, padding=1)
        gy = conv2d(image, h1, padding=1)
        g = torch.sqrt(gx**2 + gy**2)

        # avoiding division by zero error
        a = torch.atan2(gy, gx)
        a[torch.isnan(a)] = np.pi / 2  # handling the case when gx is 0

        return g, a
    def compute_quality(g1, a1, g2, a2, gF, aF, Tg, Ta, kg, ka, Dg, Da):
        G = torch.where(g1 > gF, gF / g1, torch.where(g1 == gF, gF, g1 / gF))
        A = 1 - torch.abs(a1 - aF) / (np.pi / 2)
        Qg = Tg / (1 + torch.exp(kg * (G - Dg)))
        Qa = Ta / (1 + torch.exp(ka * (A - Da)))
        return Qg * Qa
    pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
    pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)
    pF = torch.from_numpy(image_F).float().unsqueeze(0).unsqueeze(0)

    h1 = (
        torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h2 = (
        torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h3 = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    gA, aA = compute_gradients_and_orientations(pA, h1, h3)
    gB, aB = compute_gradients_and_orientations(pB, h1, h3)
    gF, aF = compute_gradients_and_orientations(pF, h1, h3)

    L = 1
    Tg, Ta = 0.9994, 0.9879
    kg, ka = -15, -22
    Dg, Da = 0.5, 0.8

    QAF = compute_quality(gA, aA, gB, aB, gF, aF, Tg, Ta, kg, ka, Dg, Da)
    QBF = compute_quality(gB, aB, gA, aA, gF, aF, Tg, Ta, kg, ka, Dg, Da)

    wA = gA**L
    wB = gB**L
    deno = torch.sum(wA + wB)
    nume = torch.sum(QAF * wA + QBF * wB)
    output = nume / deno

    return output.item()

@METRIC_REGISTRY.register()
def calculate_AG(image_F,only_gray=True,**kwargs):
    assert only_gray
    image_F = tool_adaptive_BGR2GRAY(image_F)
    r,c = image_F.shape 
    # 计算梯度
    dzdx = np.gradient(image_F, axis=1)  # 水平方向梯度
    dzdy = np.gradient(image_F, axis=0)  # 垂直方向梯度
    
    # 计算梯度幅度
    s = np.sqrt((dzdx ** 2 + dzdy ** 2) / 2)
    
    # 计算该通道的平均梯度
    # out = np.sum(s) / ((r - 1) * (c - 1))
    out = np.mean(s)
    return out

@METRIC_REGISTRY.register()
def calculate_AG_sun(image_F,**kwargs):
    image_F = tool_adaptive_BGR2GRAY(image_F)
    r,c = image_F.shape 
    Gx, Gy = np.zeros_like(image_F), np.zeros_like(image_F)

    Gx[:, 0] = image_F[:, 1] - image_F[:, 0]
    Gx[:, -1] = image_F[:, -1] - image_F[:, -2]
    Gx[:, 1:-1] = (image_F[:, 2:] - image_F[:, :-2]) / 2 # ?????

    Gy[0, :] = image_F[1, :] - image_F[0, :]
    Gy[-1, :] = image_F[-1, :] - image_F[-2, :]
    Gy[1:-1, :] = (image_F[2:, :] - image_F[:-2, :]) / 2 # ?????

    s = np.sqrt((Gx**2 + Gy**2) / 2)
    out = np.mean(s)
    # out = np.sum(s) / ((r - 1) * (c - 1))
    return out

@METRIC_REGISTRY.register()
def calculate_EN(image_F,only_gray=True,**kwargs):
    '''
    计算灰度图的熵
    '''
    assert only_gray
    image_F = tool_adaptive_BGR2GRAY(image_F)
    res = np.histogram(image_F.flatten(), range(0, 257), density=True)[0]
    res = torch.from_numpy(res)
    res = res[res != 0]
    res = torch.sum(-res * res.log2()).item()
    return res
@METRIC_REGISTRY.register()
def calculate_Qcb(image_F, image_A, image_B,resize=False,**kwargs):
    image_A = tool_adaptive_BGR2GRAY(image_A).astype(np.float64)
    image_B = tool_adaptive_BGR2GRAY(image_B).astype(np.float64)
    image_F = tool_adaptive_BGR2GRAY(image_F).astype(np.float64)
    if resize:
        image_A = tool_adaptive_resize(image_A,image_F)
        image_B = tool_adaptive_resize(image_B,image_F)
    image_A = (
        (image_A - image_A.min()) / (image_A.max() - image_A.min())
        if image_A.max() != image_A.min()
        else image_A
    )
    image_A = np.round(image_A * 255).astype(np.uint8)
    image_B = (
        (image_B - image_B.min()) / (image_B.max() - image_B.min())
        if image_B.max() != image_B.min()
        else image_B
    )
    image_B = np.round(image_B * 255).astype(np.uint8)
    image_F = (
        (image_F - image_F.min()) / (image_F.max() - image_F.min())
        if image_F.max() != image_F.min()
        else image_F
    )
    image_F = np.round(image_F * 255).astype(np.uint8)

    f0 = 15.3870
    f1 = 1.3456
    a = 0.7622
    k = 1
    h = 1
    p = 3
    q = 2
    Z = 0.0001

    M, N = image_A.shape

    # Use the correct meshgrid for frequency space
    u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
    u *= N / 30
    v *= M / 30
    r = np.sqrt(u**2 + v**2)
    Sd = np.exp(-((r / f0) ** 2)) - a * np.exp(-((r / f1) ** 2))

    # Ensure Sd matches the shape of the images
    Sd = Sd[:M, :N]  # This should ensure proper matching

    # Fourier Transform
    fused1 = np.fft.ifft2(np.fft.fft2(image_A) * Sd).real
    fused2 = np.fft.ifft2(np.fft.fft2(image_B) * Sd).real
    ffused = np.fft.ifft2(np.fft.fft2(image_F) * Sd).real

    x = np.linspace(-15, 15, 31)
    y = np.linspace(-15, 15, 31)
    X, Y = np.meshgrid(x, y)
    sigma = 2
    G1 = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    sigma = 4
    G2 = np.exp(-(X**2 + Y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    G1 = torch.from_numpy(G1).float().unsqueeze(0).unsqueeze(0)
    G2 = torch.from_numpy(G2).float().unsqueeze(0).unsqueeze(0)
    fused1 = torch.from_numpy(fused1).float().unsqueeze(0).unsqueeze(0)
    fused2 = torch.from_numpy(fused2).float().unsqueeze(0).unsqueeze(0)
    ffused = torch.from_numpy(ffused).float().unsqueeze(0).unsqueeze(0)

    buff = conv2d(fused1, G1, padding=15)
    buff1 = conv2d(fused1, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    C1P = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)
    buff = conv2d(fused2, G1, padding=15)
    buff1 = conv2d(fused2, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    C2P = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)
    buff = conv2d(ffused, G1, padding=15)
    buff1 = conv2d(ffused, G2, padding=15)
    contrast_value = buff / buff1 - 1
    contrast_value = torch.abs(contrast_value)
    CfP = (k * (contrast_value**p)) / (h * (contrast_value**q) + Z)

    mask = C1P < CfP
    Q1F = CfP / C1P
    Q1F[mask] = (C1P / CfP)[mask]
    mask = C2P < CfP
    Q2F = CfP / C2P
    Q2F[mask] = (C2P / CfP)[mask]

    ramda1 = (C1P**2) / (C1P**2 + C2P**2)
    ramda2 = (C2P**2) / (C1P**2 + C2P**2)

    Q = ramda1 * Q1F + ramda2 * Q2F
    Q = Q.mean().item()
    return Q

@METRIC_REGISTRY.register()
def calculate_Qcv(image_F, image_A, image_B,resize=False,**kwargs):
    alpha_c = 1
    alpha_s = 0.685
    f_c = 97.3227
    f_s = 12.1653

    window_size = 16
    alpha = 5

    # Preprocessing
    
    image_A = tool_adaptive_BGR2GRAY(image_A).astype(np.float64)
    image_B = tool_adaptive_BGR2GRAY(image_B).astype(np.float64)
    image_F = tool_adaptive_BGR2GRAY(image_F).astype(np.float64)
    if resize:
        image_A = tool_adaptive_resize(image_A,image_F)
        image_B = tool_adaptive_resize(image_B,image_F)
    image_A = (
        (image_A - image_A.min()) / (image_A.max() - image_A.min())
        if image_A.max() != image_A.min()
        else image_A
    )
    image_A = np.round(image_A * 255)
    image_B = (
        (image_B - image_B.min()) / (image_B.max() - image_B.min())
        if image_B.max() != image_B.min()
        else image_B
    )
    image_B = np.round(image_B * 255)
    image_F = (
        (image_F - image_F.min()) / (image_F.max() - image_F.min())
        if image_F.max() != image_F.min()
        else image_F
    )
    image_F = np.round(image_F * 255)

    h1 = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    h3 = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # Step 1: Extract Edge Information
    pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
    pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)

    img1X = conv2d(pA, h3, padding=1)
    img1Y = conv2d(pA, h1, padding=1)
    im1G = (img1X**2 + img1Y**2)**0.5

    img2X = conv2d(pB, h3, padding=1)
    img2Y = conv2d(pB, h1, padding=1)
    im2G = (img2X**2 + img2Y**2)**0.5

    M, N = image_A.shape
    ramda1 = conv2d(im1G**alpha, torch.ones(1, 1, window_size, window_size, dtype=im1G.dtype), stride=window_size)
    ramda2 = conv2d(im2G**alpha, torch.ones(1, 1, window_size, window_size, dtype=im1G.dtype), stride=window_size)

    # Similarity Measurement
    f1 = image_A - image_F
    f2 = image_B - image_F

    u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
    u *= N/8
    v *= M/8
    r = np.sqrt(u**2 + v**2)

    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))

    Df1 = np.fft.ifft2(np.fft.fft2(f1) * theta_m).real
    Df2 = np.fft.ifft2(np.fft.fft2(f2) * theta_m).real

    Df1 = torch.from_numpy(Df1).float().unsqueeze(0).unsqueeze(0)
    Df2 = torch.from_numpy(Df2).float().unsqueeze(0).unsqueeze(0)

    D1 = conv2d(Df1**2, torch.ones(1, 1, window_size, window_size, dtype=Df1.dtype)/(window_size**2), stride=window_size)
    D2 = conv2d(Df2**2, torch.ones(1, 1, window_size, window_size, dtype=Df2.dtype)/(window_size**2), stride=window_size)

    # Overall Quality
    Q = torch.sum(ramda1 * D1 + ramda2 * D2) / torch.sum(ramda1 + ramda2)

    return Q.item()

def tool_adaptive_BGR2GRAY(img):
    if len(img.shape)==3:
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        return img
    
def tool_adaptive_resize(origin,target,interpolation=cv2.INTER_CUBIC):
    H_o,W_o = origin.shape[:2]
    H_t,W_t = target.shape[:2]
    if H_o==H_t and W_o==W_t:
        return cv2.resize(origin,(W_t,H_t),interpolation=interpolation)
    else:
        return origin