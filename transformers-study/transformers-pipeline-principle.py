import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 输入文本
input_text = "我觉得很好！"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

# 分词
inputs = tokenizer(input_text, return_tensors="pt")
print(f"inputs: {inputs}")

# 用模型预测结果
outputs = model(**inputs)
print(f"outputs: {outputs}")

# 预测结果处理：归一化
sm = torch.softmax(outputs.logits, dim=-1)
print(f"sm: {sm}")

# 预测结果处理：取概率最大的项
am = torch.argmax(sm)
print(f"am: {am}")

# 预测结果处理：取最大概率的项的索引
pred = am.item()
print(f"pred: {pred}")

# 预测结果处理：将索引转换为标签
result = model.config.id2label[pred]
print(f"result: {result}")
