from transformers import AutoTokenizer, AutoModel

# 代码执行时，库会自动下载模型文件
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
