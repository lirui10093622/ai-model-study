import torch
import torch.nn as nn

# 创建一个标准的 Transformer 模型
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

# 使用它
src = torch.rand(10, 32, 512)   # (sequence_length, batch_size, d_model)
tgt = torch.rand(20, 32, 512)
out = transformer(src, tgt)     # 直接调用，内部自动完成所有计算
print(out)