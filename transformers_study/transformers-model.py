from transformers import *

name = "er/roberta-base-finetuned-ifeng-chinese"
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = name, tokenizer = tokenizer)
tokenizer = AutoTokenizer.from_pretrained(name)
result = model("今天天气不错")
print(result)