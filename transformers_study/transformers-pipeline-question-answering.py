from transformers import pipeline

# 问答类型的任务
pipe = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
result = pipe(question="中国的首都在哪里?", context="中国最大的城市是上海，中国的首都是北京")
print(result)

# 指定最大答案长度
result = pipe(question="中国的首都在哪里?", context="中国最大的城市是上海，中国的首都是北京", max_answer_len=1)
print(result)