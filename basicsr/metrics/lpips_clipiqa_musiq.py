import cv2
import numpy as np
import torch
import pyiqa
from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY

'''
采用闭包逻辑进行缓存
'''
def calculate_lpips_outer_function():
    metric = None
    @METRIC_REGISTRY.register()
    def calculate_lpips(img, img2, crop_border, input_order='HWC', **kwargs):
        '''
        用于计算LPIPS的图像
            Args:
            img (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

        Returns:
            float: PSNR result.
        关于评估指标的默认规定：img为推理图像，img2为ground truth
        默认使用numpy格式：BGR通道+uint格式
        '''

        assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = reorder_image(img, input_order=input_order)
        img2 = reorder_image(img2, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        # 计算
        # 1. 确保输入的图像是 BGR 格式，将其转换为 RGB 格式
        image1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # 2. 将图像转换为浮点型，并将值归一化到 [0, 1]
        image1_rgb = image1_rgb.astype(np.float32) / 255.0
        image2_rgb = image2_rgb.astype(np.float32) / 255.0
        
        # 3. 将 HWC 格式转换为 CHW 格式（通道优先）以适应 PyTorch 输入
        image1_rgb = np.transpose(image1_rgb, (2, 0, 1))  # 转换为 CHW 格式
        image2_rgb = np.transpose(image2_rgb, (2, 0, 1))  # 转换为 CHW 格式
        
        # 4. 转换为 PyTorch 的 Tensor，并添加 batch 维度
        image1_tensor = torch.tensor(image1_rgb).unsqueeze(0)  # shape: (1, 3, H, W)
        image2_tensor = torch.tensor(image2_rgb).unsqueeze(0)  # shape: (1, 3, H, W)
        # 采用闭包逻辑实现缓存
        nonlocal metric 
        if metric is None:
            metric = pyiqa.create_metric('lpips')
        # 6. 计算 LPIPS 得分
        lpips_score = metric(image1_tensor, image2_tensor).item()
        return lpips_score
    return calculate_lpips


def calculate_clipiqa_outer_function():
    metric = None
    @METRIC_REGISTRY.register()
    def calculate_clipiqa(img, crop_border, input_order='HWC', **kwargs):
        '''
        用于计算clipiqa的图像
            Args:
            img (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

        Returns:
            float: PSNR result.
        关于评估指标的默认规定：img为推理图像，img2为ground truth
        默认使用numpy格式：BGR通道+uint格式
        '''

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = reorder_image(img, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        # 计算
        # 1. 确保输入的图像是 BGR 格式，将其转换为 RGB 格式
        image1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 将图像转换为浮点型，并将值归一化到 [0, 1]
        image1_rgb = image1_rgb.astype(np.float32) / 255.0
        
        # 3. 将 HWC 格式转换为 CHW 格式（通道优先）以适应 PyTorch 输入
        image1_rgb = np.transpose(image1_rgb, (2, 0, 1))  # 转换为 CHW 格式
        
        # 4. 转换为 PyTorch 的 Tensor，并添加 batch 维度
        image1_tensor = torch.tensor(image1_rgb).unsqueeze(0)  # shape: (1, 3, H, W)
        # 采用闭包逻辑实现缓存
        nonlocal metric 
        if metric is None:
            metric = pyiqa.create_metric('clipiqa')
        # 6. 计算 LPIPS 得分
        clipiqa_score = metric(image1_tensor).item()
        return clipiqa_score
    return calculate_clipiqa

def calculate_musiq_outer_function():
    metric = None
    @METRIC_REGISTRY.register()
    def calculate_musiq(img, crop_border, input_order='HWC', **kwargs):
        '''
        用于计算musiq的图像
            Args:
            img (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
            input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.

        Returns:
            float: PSNR result.
        关于评估指标的默认规定：img为推理图像，img2为ground truth
        默认使用numpy格式：BGR通道+uint格式
        '''

        if input_order not in ['HWC', 'CHW']:
            raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        img = reorder_image(img, input_order=input_order)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        # 计算
        # 1. 确保输入的图像是 BGR 格式，将其转换为 RGB 格式
        image1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 将图像转换为浮点型，并将值归一化到 [0, 1]
        image1_rgb = image1_rgb.astype(np.float32) / 255.0
        
        # 3. 将 HWC 格式转换为 CHW 格式（通道优先）以适应 PyTorch 输入
        image1_rgb = np.transpose(image1_rgb, (2, 0, 1))  # 转换为 CHW 格式
        
        # 4. 转换为 PyTorch 的 Tensor，并添加 batch 维度
        image1_tensor = torch.tensor(image1_rgb).unsqueeze(0)  # shape: (1, 3, H, W)
        # 采用闭包逻辑实现缓存
        nonlocal metric 
        if metric is None:
            metric = pyiqa.create_metric('musiq')
        # 6. 计算 LPIPS 得分
        musiq_score = metric(image1_tensor).item()
        return musiq_score
    return calculate_musiq
