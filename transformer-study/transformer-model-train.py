import torch
import torch.nn as nn
import math

# ==================== 1. 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]


# ==================== 2. 完整 Seq2Seq Transformer ====================
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """src, tgt: (seq_len, batch)"""
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)  # (seq_len, batch, vocab_size)


# ==================== 3. 任务：序列反转 ====================
def generate_batch(batch_size, seq_len, vocab_size):
    """生成一批数据：输入序列，目标是反转后的序列"""
    # 序列中每个 token 取值范围 [1, vocab_size-1]，0 保留给 <start>/<pad>
    src = torch.randint(1, vocab_size, (seq_len, batch_size))
    tgt = torch.flip(src, dims=[0])  # 目标：反转

    # Teacher forcing：decoder 输入 = 目标序列去掉最后一个 token，前面补 <start>(0)
    tgt_input = torch.cat([torch.zeros(1, batch_size, dtype=torch.long), tgt[:-1]], dim=0)
    return src, tgt_input, tgt


def translate(model, src, max_len=20, start_token=0):
    """用训练好的模型做推理（贪心解码）"""
    model.eval()
    batch_size = src.size(1)
    device = next(model.parameters()).device

    # encoder 只跑一次
    src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb)

    # decoder 自回归
    tgt = torch.full((1, batch_size), start_token, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_emb = model.pos_encoder(model.embedding(tgt) * math.sqrt(model.d_model))
        output = model.transformer.decoder(tgt_emb, memory)
        logits = model.fc_out(output[-1:, :, :])  # 只看最后一个时间步
        pred = logits.argmax(dim=-1)               # (1, batch)
        tgt = torch.cat([tgt, pred], dim=0)
    return tgt[1:]  # 去掉开头的 <start>


# ==================== 4. 训练 ====================
def train():
    # 超参数（用小模型快速演示）
    vocab_size = 50    # 词汇量
    d_model = 128      # 嵌入维度
    nhead = 4          # 注意力头数
    num_layers = 3     # encoder/decoder 层数
    seq_len = 8        # 序列长度
    batch_size = 64
    num_epochs = 30
    batches_per_epoch = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    model = TransformerSeq2Seq(vocab_size, d_model, nhead, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')
    print(f'\n开始训练（任务：序列反转，序列长度={seq_len}）...\n')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _ in range(batches_per_epoch):
            src, tgt_input, tgt = generate_batch(batch_size, seq_len, vocab_size)
            src, tgt_input, tgt = src.to(device), tgt_input.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt_input)                    # (seq_len, batch, vocab_size)
            loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / batches_per_epoch

        # 每 5 个 epoch 展示一次推理效果
        if (epoch + 1) % 5 == 0:
            test_src = torch.randint(1, vocab_size, (seq_len, 3), device=device)
            with torch.no_grad():
                pred = translate(model, test_src, max_len=seq_len)
            print(f'Epoch {epoch+1:2d}, Loss: {avg_loss:.4f}')
            for i in range(min(3, test_src.size(1))):
                src_str = ' '.join(str(x.item()) for x in test_src[:, i])
                pred_str = ' '.join(str(x.item()) for x in pred[:, i])
                tgt_str = ' '.join(str(x.item()) for x in torch.flip(test_src[:, i], dims=[0]))
                print(f'  输入: [{src_str}]')
                print(f'  目标: [{tgt_str}]')
                print(f'  预测: [{pred_str}]')
                print()
        else:
            print(f'Epoch {epoch+1:2d}, Loss: {avg_loss:.4f}')

    return model


# ==================== 5. 最终测试 ====================
if __name__ == '__main__':
    model = train()

    print('=' * 40)
    print('训练完成！最终效果测试:')
    print('=' * 40)

    device = next(model.parameters()).device
    vocab_size = 50
    seq_len = 8

    model.eval()
    test_src = torch.randint(1, vocab_size, (seq_len, 5), device=device)
    with torch.no_grad():
        pred = translate(model, test_src, max_len=seq_len)

    correct = 0
    total = test_src.size(0) * test_src.size(1)
    for i in range(test_src.size(1)):
        src_seq = test_src[:, i]
        tgt_seq = torch.flip(src_seq, dims=[0])
        pred_seq = pred[:, i]
        is_correct = torch.equal(tgt_seq, pred_seq)
        if is_correct:
            correct += test_src.size(0)
        print(f'输入: {[x.item() for x in src_seq]}')
        print(f'目标: {[x.item() for x in tgt_seq]}')
        print(f'预测: {[x.item() for x in pred_seq]}')
        print(f'结果: {"✓" if is_correct else "✗"}')
        print()

    print(f'准确率: {correct}/{total} ({correct/total*100:.1f}%)')