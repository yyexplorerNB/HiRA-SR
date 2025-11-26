import torch
import numpy as np
import os
from PIL import Image
import random

def rename_all_files_from_OmniSR(directory):
    keywords = ['_HR_x2','_HR_x3','_HR_x4','_HR_x8',
                '_LRBI_x2','_LRBI_x3','_LRBI_x4','_LRBI_x8',
                ]
    for root, dirs, files in os.walk(directory):
        for file in files:
            for keyword in keywords:
                if keyword in file:
                    new_file_name = file.replace(keyword, '')
                    os.rename(os.path.join(root, file), os.path.join(root, new_file_name))

def use_bicubic_to_upscale_images(input_folder, output_folder,upscale=4,suffix='_bicubic'):
    for root, dirs, files in os.walk(input_folder):
        # 对于找到的每个文件和目录
        for name in files:
            # 检查文件是否为图片
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                base_name,extension = os.path.splitext(name)
                
                file_path = os.path.join(root, name)
                # 读取图片
                with Image.open(file_path) as img:
                    # 计算新的尺寸
                    new_size = (img.width * upscale, img.height * upscale)
                    # 使用双三次插值方法上采样
                    img_upscaled = img.resize(new_size, Image.BICUBIC)
                    # 构建输出路径，保持原始文件结构
                    name = base_name+suffix+extension
                    output_file_path = os.path.join(root, name)
                    output_path = os.path.join(output_folder, os.path.relpath(output_file_path, input_folder))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    # 保存图片
                    img_upscaled.save(output_path)

def set_seed(seed=0):
    '''
    设置随机种子，这需要在所有导入包的后面，因为basicsr默认重置随机种子
    '''
    random.seed()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
        torch.backends.cudnn.deterministic = True