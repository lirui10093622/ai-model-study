from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型和分词器
local_model_path = 'C:\\Users\\NLDS\\.cache\\huggingface\\hub\models--uer--roberta-base-finetuned-ifeng-chinese\\snapshots\\eaf91be1a19b4a37db517a2da182fbbf7fe0ad27'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

# 输入新闻文本
news = "特斯拉发布新款电动皮卡，股价盘中大涨"
inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True)

# 进行预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = predictions.argmax().item()

# 查看结果
print(f"新闻类别ID: {predicted_class_id}")
print(f"各类别概率: {predictions}")