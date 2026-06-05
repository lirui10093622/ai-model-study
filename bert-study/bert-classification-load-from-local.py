import logging

from transformers import pipeline

# 1. 确定任务类型，并选择合适的模型(本地加载)
local_model_path = 'C:\\Users\\NLDS\\.cache\\huggingface\\hub\\models--bert-base-chinese\\snapshots\\8f23c25b06e129b6c986331a13d8d025a92cf0ea'
classifier = pipeline("text-classification", model=local_model_path)

# 2. 输入文本
text = "今天天气真糟糕"

# 3. 进行推理
result = classifier(text)

logging.info(f"result: {result}")

label_map = {"LABEL_0": "负面", "LABEL_1": "正面"}

label = label_map[result[0]['label']]

score = result[0]['score']

print(f"{label}: {score}")
