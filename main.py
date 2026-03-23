import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 生成数据：y = 2*x1 + 3*x2 + 1.5*x3 + 1 + 噪声
np.random.seed(42)
N = 200
X = np.random.randn(N, 3)            # 200个样本，每个样本3个特征
y = 2*X[:,0] + 3*X[:,1] + 1.5*X[:,2] + 1 + np.random.randn(N)*0.1

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32)