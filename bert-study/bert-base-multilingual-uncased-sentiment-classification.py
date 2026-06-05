from transformers import pipeline

classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

result = classifier("这部电影还不错")
print(result)
# 输出类似: [{'label': '3 stars', 'score': 0.78}]