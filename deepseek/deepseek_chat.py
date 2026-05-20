import json
import deepseek_api

question = input("Q: ")
while True:
  resp = deepseek_api.call_deepseek_v4_pro_with_question(question)
  # 读取响应中的内容
  response_json = json.loads(resp.content)
  answer = response_json["choices"][0]["message"]["content"]
  print(f"A: {answer}")
  question = input("Q: ")
