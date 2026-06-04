from transformers import *

name = "er/roberta-base-finetuned-ifeng-chinese"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = name, tokenizer = tokenizer)
result = model("今天天气不错")
print(result)