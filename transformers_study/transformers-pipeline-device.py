import time

import torch
from transformers import pipeline

def print_time(pipe):
    times = []
    for i in range(1000):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        pipe("今天天气不错")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print(sum(times) / 1000)

# 指定设备为CPU
pipe = pipeline("text-classification", device='cpu')
print_time(pipe)

# 指定设备为GPU
pipe = pipeline("text-classification", device=0)
print_time(pipe)