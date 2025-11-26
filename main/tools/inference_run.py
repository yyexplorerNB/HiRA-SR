import torch
from pathlib import Path
import sys
sys.path.append('./main') # ./main
sys.path.append('./')
from architectures.HiRA_SR_arch import HiRA_SR as NET
import cv2
import numpy as np
import time
from utils.utils import set_seed
from utils.inference import inference

def show():
    '''
    用于针对单张图片进行超分并展示。
    '''
    weight_path = 'weights/train_RealESR_HiRA-SR_DF2K_x2_net_g_latest.pth'
    input_path = r"C:\Users\yy_ex\Desktop\20250924155226_58_168.jpg"
    output_path = r"C:\Users\yy_ex\Desktop\20250924155226_58_168_SR.png"
    upscale = 2
    net = NET(
        upscale= 2,
        in_chans= 3,
        img_size= 64,
        window_size= 8,
        img_range= 1.,
        depths= [6,6,6,6],
        embed_dim= 60,
        num_heads= [6,6,6,6],
        mlp_ratio= 2,
        upsampler= 'pixelshuffledirect',
        )
    loadnet = torch.load(weight_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    net.load_state_dict(loadnet[keyname], strict=True)
    set_seed(42)
    infer = inference(input_path=input_path, # 如果为混合文件模式，则为data_root路径。如果为单个文件，则为文件路径
                    output_path=output_path, # 如果为混合文件模式，则为data_root路径。如果为单个文件，则为文件路径。如果为none则不输出任何文件。
                    upscale=upscale, # 如果是去雨去雾则为1，超分则为2,3,4等
                    resize=None, 
                    )
    infer.init_net(net=net)
    infer.inference(show=False)

if __name__ == '__main__':
    show()
    cv2.waitKey(0)