import requests
import os
import json

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = "sk-c59a457d22ae4b0da273eae952b01d18"

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}


def call_deepseek_chat():
  resp = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json={
      "model": "deepseek-chat",
      "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
      ],
      "stream": False
  })

  print(f"status code: {resp.status_code}")
  print(f"response: {resp.content}")

def call_deepseek_v4_pro():
  resp = requests.post("https://api.deepseek.com/chat/completions", headers = headers, json={
        "model": "deepseek-v4-pro",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "stream": False
      })
  print(f"status code: {resp.status_code}")
  print(f"response: {json.dumps(json.loads(resp.content), indent=4, ensure_ascii=False)}")

call_deepseek_v4_pro()