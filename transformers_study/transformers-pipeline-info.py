from transformers.pipelines import SUPPORTED_TASKS

# 支持的任务类型
for k, v in SUPPORTED_TASKS.items():
    print(f"{k} -> {v}")
