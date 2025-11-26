from pathlib import Path
import sys
import torch
import cv2
import numpy as np
import time
import os
from tqdm import tqdm
from copy import deepcopy

class inference():
    '''
    用于对图像恢复算法（去雨、去雾、超分等）进行正向推理，支持的格式有视频、图片。

    single_video模式和single_image模式：输入路径和输出路径均为文件路径

    files模式：输入路径和输出路径为文件夹。程序会自动扫描文件夹内的所有子树，并以相同的树结构对所有视频和图片进行处理并输出。

    upscale: 如果是去雨去雾则为1，超分则为2,3,4等

    resize: [w, h] or None 在输出时强制进行resize

    以pytorch为推理框架
    '''
    def __init__(self,
                 input_path : str, # 如果为混合文件模式，则为data_root路径。如果为单个文件，则为文件路径
                 output_path : str, # 如果为混合文件模式，则为data_root路径。如果为单个文件，则为文件路径。如果为none则不输出任何文件。
                #  mode : str, # in ['files','single_video','single_image']
                 upscale : int , # 如果是去雨去雾则为1，超分则为2,3,4等
                 resize : list = None, # 是否在输出时强制进行resize
                 waitKey : bool = False, # 推理完成时是否等待操作
                 use_seq : bool = False, # 是否使用视频序列模式
                 ) -> None:
        # assert mode in ['files','single_video','single_image']
        self.input_path = input_path
        self.output_path = output_path
        self.upscale = upscale
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.supported_formats = [('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'), # 图片
                                  ('.mp4', '.mov', '.avi')] # 视频
        self.mode = self._verify_mode(input_path,output_path)
        self.resize = resize
        self.waitKey = waitKey
        self.use_seq = use_seq

    def inference(self,show=True):
        timer_start = time.time()
        assert self.net is not None , '需要初始化网络模型'
        try: 
            window_size = self.net.window_size
        except:
            window_size = 1
        if self.mode == 'single_video':
            cap = cv2.VideoCapture(self.input_path)
            self._inference_single_video(cap=cap,output_file=self.output_path,window_size=window_size,show=show)
            cap.release()
        elif self.mode == 'single_image':
            img = cv2.imread(self.input_path,cv2.IMREAD_COLOR)
            img,_ = self._inference_single_image(img=img,window_size=window_size,show=show,output_file=self.output_path)
        elif self.mode == 'files': # 对文件夹内文件树内的所有视频和图片进行处理
            pbar = tqdm()
            for root,dirs,files in os.walk(self.input_path):
                for name in files:
                    pbar.set_description(os.path.join(root,name))
                    # 检查文件是否为图片
                    if name.lower().endswith(self.supported_formats[0]):
                        base_name,extension = os.path.splitext(name)
                        file_path = os.path.join(root, name)
                        if self.output_path is not None:
                            # 计算输出的文件路径
                            output_file_path = os.path.join(self.output_path, os.path.relpath(file_path, self.input_path))
                            # 创建文件夹
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        else:
                            output_file_path=None
                        # 图片推理
                        img = cv2.imread(file_path,cv2.IMREAD_COLOR)
                        img,_ = self._inference_single_image(img=img,window_size=window_size,show=show,
                                                             output_file=output_file_path)
                    # 检查文件是否为视频
                    if name.lower().endswith(self.supported_formats[1]):
                        base_name,extension = os.path.splitext(name)
                        file_path = os.path.join(root, name)
                        if self.output_path is not None:
                            output_file_path = os.path.join(self.output_path, os.path.relpath(file_path, self.input_path))
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        else:
                            output_file_path = None
                        cap = cv2.VideoCapture(file_path)
                        self._inference_single_video(cap=cap,output_file=output_file_path,window_size=window_size,show=show)
                        cap.release()
                    pbar.update(1)
        print('\n finish !')
        if self.waitKey :
            cv2.waitKey(0)
        return

    def init_net(self,net,type='pytorch'):
        '''
        如果网络模型基于Vit, 需要确保这个网络模型的实例中存在window_size (int)变量。
        模型的输入需要是[b,c,h,w]或者时序的[b,t,c,h,w],并且通道需要是RGB
        '''
        assert type in ['pytorch','rknn']
        if type == 'pytorch':
            self.net = net.to(self.device)
        elif type == 'rknn':
            self.net = net
        return
    def _inference_single_video(self,cap,output_file=None,window_size=None,show=True):
        '''
        cap : cv2.VideoCapture
        output_file : 输出视频的路径 or none

        output:
                    videowriter  cv2.VideoWriter
                    timer secends
        '''
        assert cap.isOpened()
        timer_start = time.time()
        # 获取视频的帧率和帧的尺寸
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # 定义视频编码和创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        if self.resize is None:
            videowriter = cv2.VideoWriter(output_file, fourcc, video_fps, (frame_width*self.upscale, frame_height*self.upscale)) if output_file is not None else None
        else:
            videowriter = cv2.VideoWriter(output_file, fourcc, video_fps, self.resize) if output_file is not None else None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                img,_=self._inference_single_image(img=frame,
                                             window_size=window_size,
                                             show=show,output_file=None)
            except:
                break
            if videowriter is not None: videowriter.write(img)
        if videowriter is not None: videowriter.release()
        
        return videowriter , time.time()-timer_start
    def _inference_single_image(self,img,window_size=None,show=True,output_file=None):
        '''
        当模型为vit时，需要进行pad
        img : opencv or numpy [h,w,BGR]
        window_size : 窗口大小  int or none
        output_file : 输出图片的路径 or none
        output :    numpy [h,w,BGR]
                    timer secends
        '''
        if window_size is None:
            window_size = 1
        img_for_show = deepcopy(img)
        
        
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().to(device=self.device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            # padding
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = torch.nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            # inference yyNet
            timer_start = time.time()
            if self.use_seq : img = img.unsqueeze(1)
            output_swinir = self.net(img)
            h, w = output_swinir.size()[-2:]
            output_swinir = output_swinir[:, :, 0:h - mod_pad_h * self.upscale, 0:w - mod_pad_w * self.upscale]
            output_swinir = output_swinir.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            used_time = time.time() - timer_start
            # show image
            output_swinir = np.transpose(output_swinir[[2, 1, 0], :, :], (1, 2, 0))
            output_swinir = (output_swinir * 255.0).round().astype(np.uint8)
        if self.resize is not None:
            output_swinir = cv2.resize(output_swinir,self.resize,cv2.INTER_LINEAR)
        if show:
            cv2.imshow('output_image',output_swinir)
            cv2.imshow('input_image',img_for_show)
            # 显示bicubic的结果
            bicu_h,bicu_w = img_for_show.shape[:2]
            cv2.imshow('bicubic_image',cv2.resize(img_for_show, (bicu_w*self.upscale, bicu_h*self.upscale), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) == ord('q'):
                return None
        if output_file is not None:
            cv2.imwrite(output_file,output_swinir)
        print(f'Image running time:{used_time:.4f}s')
        return output_swinir,used_time
    
    def _verify_mode(self,input_path,output_path):
        in_mode = None
        if os.path.isfile(input_path):
            if input_path.lower().endswith(self.supported_formats[0]):
                in_mode = 'single_image'
            elif input_path.lower().endswith(self.supported_formats[1]):
                in_mode = 'single_video'
        elif os.path.isdir(input_path):
            in_mode = 'files'
        if in_mode is not None:
            return in_mode
        else:
            raise ValueError("Path Error !! fix it immediately !! or ask YingYuan for help !! GoGoGo !!")

