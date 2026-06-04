from transformers import BertTokenizer, BertForSequenceClassification
import torch
# 指定模型文件夹的路径
model_path = 'C:\\Users\\NLDS\\.cache\\huggingface\\hub\\models--bert-base-chinese\\snapshots\\8f23c25b06e129b6c986331a13d8d025a92cf0ea'
# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
# 准备输入数据
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt')
# 在模型上进行推理
outputs = model(**inputs)
# 获取预测结果
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)