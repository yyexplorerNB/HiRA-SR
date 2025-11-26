import numpy as np
import random
import torch
from scipy.spatial.distance import cosine
class CalculateRankPerLayer():
    def __init__(self) -> None:
        self.count_rank = 0
        self.rank_per_layer=[]
        self.count_entropy = 0
        self.entropy_per_layer=[]
    
    def calculate_rank(self,mat):
        '''
        输入：mat: torch.tensor [b_,n,n]
        '''
        mat = mat.cpu().numpy()
        
        rank = np.linalg.matrix_rank(mat).mean()
        self.rank_per_layer.append(rank)
        self.count_rank+=1
        print(f'第{self.count_rank}层的所有注意力图的平均秩为:{rank:.02f}')

    def calculate_average_entropy(self,mat):
        """
        计算输入三维张量中每个矩阵的信息熵，并返回平均信息熵。
        
        参数:
            mat (torch.Tensor): 形状为 (batch, n, n) 的三维张量，其中第0个维度表示 batch 个矩阵。

        返回:
            float: 所有矩阵的信息熵的平均值。
        """
        # 确保输入是三维张量
        assert mat.dim() == 3, "输入必须是一个三维张量 (batch, n, n)"
        
        # 获取batch的数量
        batch_size = mat.size(0)
        
        # 初始化熵总和
        entropy_sum = 0.0
        
        # 对每个矩阵计算熵
        for i in range(batch_size):
            matrix = mat[i]
            
            # 将矩阵中的元素归一化为概率分布
            total_sum = torch.sum(matrix)
            # 避免除零错误
            if total_sum == 0:
                continue
            prob_matrix = matrix / total_sum
            
            # 使用掩码过滤掉 0 值，避免 log(0) 错误
            mask = prob_matrix > 0
            filtered_prob = prob_matrix[mask]
            
            # 计算当前矩阵的信息熵
            entropy = -torch.sum(filtered_prob * torch.log2(filtered_prob))
            entropy_sum += entropy.item()
        
        # 返回平均熵
        average_entropy = entropy_sum / batch_size if batch_size > 0 else 0.0
        self.count_entropy+=1
        self.entropy_per_layer.append(average_entropy)
        print(f'第{self.count_entropy}层的所有注意力图的平均信息熵为:{average_entropy:.03f}')
        return average_entropy

class CalculateCosSimilarityPerLayer():
    def __init__(self,) -> None:
        self.data = []
        self.count = 0
    def gather(self,mat):
        self.count += 1 
        mat = mat.cpu().numpy()
        b_,h,n_,n__ = mat.shape
        sim = np.zeros(shape=(b_))
        for window in range(b_):
            mat_similarity = mat[window,:,:,:]
            vectors = mat_similarity.reshape(h, -1)
            cosine_similarities = np.zeros((h, h))
            for i in range(h):
                for j in range(h):
                    cosine_similarities[i, j] = 1 - cosine(vectors[i], vectors[j])
            # print(cosine_similarities)
            sim[window] = cosine_similarities.mean()
        self.data.append(sim)
        print(f'第{self.count}层的不同头相似度为：{sim.mean()}')

    def get_avg_sim(self):
        mat = np.array(self.data)
        return mat.mean()
    def norm_linear(self,mat):
        max = mat.max()
        min = mat.min()
        return (mat-min)/(max-min)
    
def low_rank_approximation(tensor: torch.Tensor, rank: int,epsilon = 1e-6) -> torch.Tensor:
    """
    对 batch 个矩阵进行低秩近似的并行实现
    使用随机SVD对 batch 个矩阵进行快速低秩近似，并处理秩小于目标 rank 的情况

    参数:
        tensor: 形状为 (batch, a, b) 的 pytorch 张量，其中 batch 表示有多个矩阵
        rank: 目标秩，用于低秩近似

    返回:
        形状为 (batch, a, b) 的 pytorch 张量，其中每个矩阵都经过了低秩近似
    """
    shape = tensor.shape
    tensor = tensor.reshape(-1,shape[-2],shape[-1])
    with torch.no_grad():
        approx_tensor = randomized_svd_safe(tensor,rank)
    return approx_tensor.reshape(*shape)

def truncated_svd_safe(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """
    使用截断SVD对 batch 个矩阵进行低秩近似，并处理秩小于目标 rank 的情况

    参数:
        tensor: 形状为 (batch, a, b) 的 pytorch 张量
        rank: 目标秩

    返回:
        形状为 (batch, a, b) 的 pytorch 张量，每个矩阵经过低秩近似
    """
    batch, a, b = tensor.shape
    
    # 对所有矩阵执行SVD, U: (batch, a, a), S: (batch, min(a,b)), Vh: (batch, b, b)
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
    
    # 处理秩小于目标rank的情况，计算每个batch的实际秩
    actual_rank = min(S.size(1), rank)  # 选择不超过 min(a, b) 的 rank

    # 仅保留前 actual_rank 个奇异值和对应的奇异向量
    U_k = U[:, :, :actual_rank]  # (batch, a, actual_rank)
    S_k = S[:, :actual_rank]     # (batch, actual_rank)
    Vh_k = Vh[:, :actual_rank, :]  # (batch, actual_rank, b)

    # 构造对角矩阵 S_k, 形状为 (batch, actual_rank, actual_rank)
    S_k_diag = torch.diag_embed(S_k)

    # 构建近似矩阵 A_k = U_k * S_k * V_k^T, 形状为 (batch, a, b)
    approx_tensor = torch.matmul(torch.matmul(U_k, S_k_diag), Vh_k)

    return approx_tensor

def randomized_svd_safe(tensor: torch.Tensor, rank: int, n_iter: int = 5) -> torch.Tensor:
    """
    使用随机SVD对 batch 个矩阵进行快速低秩近似，并处理秩小于目标 rank 的情况

    参数:
        tensor: 形状为 (batch, a, b) 的 pytorch 张量
        rank: 目标秩
        n_iter: 随机投影的迭代次数，越大精度越高，计算时间也会增加

    返回:
        形状为 (batch, a, b) 的 pytorch 张量，每个矩阵经过快速低秩近似
    """
    batch, a, b = tensor.shape

    # 创建随机投影矩阵，形状为 (batch, b, rank)
    random_matrix = torch.randn(batch, b, rank, device=tensor.device)

    # 计算投影 Y = A @ Ω，形状为 (batch, a, rank)
    Y = torch.matmul(tensor, random_matrix)

    # 使用QR分解对 Y 进行正交化，形状为 (batch, a, rank)
    Q, _ = torch.linalg.qr(Y)

    # 计算 B = Q^T @ A，形状为 (batch, rank, b)
    B = torch.matmul(Q.transpose(-2, -1), tensor)

    # 对 B 执行 SVD，得到形状为 (batch, rank, rank) 和 (batch, rank, b)
    U_b, S_b, Vh_b = torch.svd(B)

    # 处理秩小于目标rank的情况
    actual_rank = min(S_b.size(1), rank)  # 保证不会超过实际的秩

    # 取前 actual_rank 个奇异值及其对应的 U 和 V 向量
    U_k = torch.matmul(Q, U_b[:, :, :actual_rank])  # (batch, a, actual_rank)
    S_k = S_b[:, :actual_rank]                      # (batch, actual_rank)
    Vh_k = Vh_b[:, :, :actual_rank]                 # (batch, b, actual_rank)

    # 构造对角矩阵 S_k，形状为 (batch, actual_rank, actual_rank)
    S_k_diag = torch.diag_embed(S_k)

    # 构建近似矩阵 A_k = U_k * S_k * V_k^T，形状为 (batch, a, b)
    approx_tensor = torch.matmul(torch.matmul(U_k, S_k_diag), Vh_k.transpose(-2, -1))

    return approx_tensor