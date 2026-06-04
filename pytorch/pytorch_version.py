import torch
print(torch.__version__)          # 例如 '2.0.1+cpu' 表示 CPU 版本
print(torch.cuda.is_available())  # 如果返回 False，说明无法使用 CUDA
