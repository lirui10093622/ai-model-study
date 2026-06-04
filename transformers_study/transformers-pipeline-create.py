from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 指定任务类型
print("开始创建管道：指定任务类型")
pipeline("text-classification")

# 指定模型
print("开始创建管道：指定模型")
pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")

# 指定模型和分词器
print("开始创建管道：指定模型和分词器")
pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese", tokenizer="uer/roberta-base-finetuned-dianping-chinese")

# 预先加载模型和分词器
print("开始创建管道：预先加载模型和分词器")
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipeline("text-classification", model=model, tokenizer=tokenizer)