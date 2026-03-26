import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成合成数据：y = 2*x + 1 + 噪声
np.random.seed(42)
torch.manual_seed(42)

x = np.random.rand(100, 1) * 10          # x 在 [0,10] 之间
y = 2 * x + 1 + np.random.randn(100, 1) * 0.5   # 真实函数 + 噪声

# 转换为 PyTorch Tensor
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 2. 定义模型：一个神经元（单层线性）
class OneNeuronModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个线性层：输入维度 1，输出维度 1，包含 w 和 b
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = OneNeuronModel()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()          # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)   # 随机梯度下降

# 4. 训练循环
epochs = 500
losses = []

for epoch in range(epochs):
    # 前向传播
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)

    # 反向传播
    optimizer.zero_grad()   # 清空上一步的梯度
    loss.backward()         # 计算梯度
    optimizer.step()        # 更新参数

    losses.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 训练后查看学到的参数
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f'\n训练后的参数: w = {w:.4f}, b = {b:.4f} (真实值 w=2, b=1)')

# 6. 可视化训练曲线和拟合结果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')

plt.subplot(1, 2, 2)
plt.scatter(x, y, label='真实数据', alpha=0.6)
x_line = np.linspace(0, 10, 100)
y_line = w * x_line + b
plt.plot(x_line, y_line, 'r-', label=f'拟合: y = {w:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('拟合效果')

plt.tight_layout()
plt.show()