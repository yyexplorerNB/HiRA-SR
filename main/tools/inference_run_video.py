import torch
from pathlib import Path
import sys
sys.path.append('./main') # ./main
from architectures.yyNet_realtime_arch import yyNet_realtime as Net
import cv2
import numpy as np
import time
from utils.utils import set_seed
from utils.inference import inference

def show():
    '''
    用于针对单张图片进行超分并展示。
    '''
    weight_path = 'weights/yyNet_realtime_init.pth'
    input_path = r"datasets/origin_LR_video/all_72P.mov"
    output_path = None
    upscale = 4
    net = Net(
        in_channels=3,
        mid_channels=32,
        propagation_blocks=3,
        RRG_blocks=1,
        RRG_MRBs=1,
        use_cpu_cache=True,
        upscale=4,
        history=2,
        spynet_path=None,
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
                    use_seq=True,
                    )
    infer.init_net(net=net)
    infer.inference(show=True)

if __name__ == '__main__':
    show()
    cv2.waitKey(0)