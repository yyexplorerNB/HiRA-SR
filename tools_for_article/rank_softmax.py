import numpy as np
import torch

batch = 256
x = torch.randn(size=[batch,64,10])
y = torch.randn(size=[batch,10,64])
mat = x @ y
# mat = torch.tensor([
#     [1,2],
#     [2,4]
# ])
soft = torch.nn.Softmax(-1)
mat = soft(mat)
mat = mat.cpu().numpy()
rank = np.linalg.matrix_rank(mat)
print(rank.max(),rank.min())