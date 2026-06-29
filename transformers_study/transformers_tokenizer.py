from transformers import AutoTokenizer

name = "uer/roberta-base-finetuned-ifeng-chinese"
tokenizer = AutoTokenizer.from_pretrained(name)

print(tokenizer.vocab)
