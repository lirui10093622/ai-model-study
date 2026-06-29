from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

name = "uer/roberta-base-finetuned-ifeng-chinese"

tokenizer = AutoTokenizer.from_pretrained(name)

model = AutoModelForSequenceClassification.from_pretrained(name)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = pipe("今天天气不错")

print(result)