'''
请你使用python帮我解决如下问题：
这是一个计算机视觉领域的关于计算图像质量指标的问题。现在有代表Ground Truth的文件夹GT_path和代表模型推理结果的文件夹results_path，他们的文件树结构完全相同，即：'GT_path/{数据集名称}/*.png'，其中{数据集名称}代表数据集名称，*.png代表数据集下的图片。现在我需要你帮我统计模型在不同数据集下的LPIPS指标。
其他要求如下：
1. 使用opencv读取图片。使用pyiqa计算结果。
'''
import torch
import os
import cv2
import numpy as np
from pyiqa import create_metric
from tqdm import tqdm
import re

def calculate_lpips(GT_path, results_path, results_suffix=''):
    # 创建 LPIPS 评估器
    lpips_metric = create_metric('lpips')

    # 初始化结果存储字典
    lpips_scores = {}

    # 遍历每个数据集
    for dataset_name in os.listdir(GT_path):
        gt_dataset_path = os.path.join(GT_path, dataset_name)
        results_dataset_path = os.path.join(results_path, dataset_name)

        # 确保路径有效
        if not os.path.isdir(gt_dataset_path) or not os.path.isdir(results_dataset_path):
            continue

        # 获取当前数据集下的所有 Ground Truth 图片文件名
        gt_images = sorted([f for f in os.listdir(gt_dataset_path) if f.endswith('.png')])

        # 获取对应模型结果的图片文件名
        results_images = sorted([f for f in os.listdir(results_dataset_path) if re.search(f"{results_suffix}\\.png$", f)])

        # 提取图片的核心名称用于匹配（去掉后缀）
        gt_image_core_names = [os.path.splitext(f)[0] for f in gt_images]
        results_image_core_names = [re.sub(f"{results_suffix}", "", os.path.splitext(f)[0]) for f in results_images]

        # 确保图片核心名称匹配
        assert gt_image_core_names == results_image_core_names, f"File names in {dataset_name} do not match."

        # 计算当前数据集的 LPIPS 指标
        dataset_scores = []
        for gt_image_name, result_image_name in tqdm(zip(gt_images, results_images), desc=f"Processing dataset: {dataset_name}"):
            gt_image_path = os.path.join(gt_dataset_path, gt_image_name)
            result_image_path = os.path.join(results_dataset_path, result_image_name)

            # 读取图像 (BGR -> RGB)
            gt_image = cv2.imread(gt_image_path)
            result_image = cv2.imread(result_image_path)

            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            # 转换为 Tensor 格式 (C, H, W), 归一化到 [0, 1]
            gt_image_tensor = np.transpose(gt_image / 255.0, (2, 0, 1))
            result_image_tensor = np.transpose(result_image / 255.0, (2, 0, 1))

            # 确保类型为 float32
            score = lpips_metric(
                torch.tensor(gt_image_tensor, dtype=torch.float32).unsqueeze(0),
                torch.tensor(result_image_tensor, dtype=torch.float32).unsqueeze(0)
            )
            dataset_scores.append(score.item())

        # 存储当前数据集的平均 LPIPS
        lpips_scores[dataset_name] = np.mean(dataset_scores)

    return lpips_scores

if __name__ == "__main__":

    # 指定 Ground Truth 和推理结果文件夹路径
    GT_path = "tools_for_YYYYDS_article/visual comparison/Ground_Truth_x4"
    results_path = "tools_for_YYYYDS_article/visual comparison/EDSR_x4"
    suffix = ''

    # 计算 LPIPS 指标
    lpips_results = calculate_lpips(GT_path, results_path,suffix)

    # 打印结果
    for dataset, score in lpips_results.items():
        print(f"Dataset: {dataset}, Average LPIPS: {score:.4f}")
