from transformers import AutoTokenizer

name = "uer/roberta-base-finetuned-ifeng-chinese"

tokenizer = AutoTokenizer.from_pretrained(name)

# 输入文本
text = "今天天气不错"

# 分词
tokens = tokenizer.tokenize(text)
print(tokens)

# 转换为Token IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

for id in ids:
    print(tokenizer.vocab[id])

# Token IDs转为Token
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)

# 解密
input_ids = tokenizer.encode(text)
print(input_ids)

# 输入文本转为字典
tokens = tokenizer.decode(input_ids)
print(tokens)

# 分词+转换为ID+其他处理
input_ids = tokenizer(text)
print(input_ids)

# 逆向分词
dec = tokenizer.decode(input_ids["input_ids"])
print(dec)

ids = tokenizer.encode(text, padding="max_length", max_length=15)
print(ids)

ids = tokenizer.encode(text, padding="max_length", max_length=15, truncation=True)
print(ids)