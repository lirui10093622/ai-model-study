import json

from deepseek import deepseek_api

from weather_api import get_weather_api

key = "7815c66716eeb680e1227c195fa9aec8"

def get_weather (location):
    return get_weather_api(location)

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取指定位置的当前天气信息",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "城市名称，例如：上海、北京",
          }
        },
        "required": ["location"],
      }
    },
  }
]

question = input("Q: ")

resp = deepseek_api.call_deepseek_v4_pro_with_question(question, tools)

# 读取响应中的内容
response_json = json.loads(resp.content)
response_message = response_json["choices"][0]["message"]
if "tool_calls" in response_message:
  tool_call = response_message["tool_calls"][0]
  function_name = tool_call["function"]["name"]
  function_args = json.loads(tool_call["function"]["arguments"])

  if function_name == "get_weather":
    city_name = function_args.get("location")
    # 调用我们自己的函数获取真实天气数据
    weather = get_weather(city_name)
    print(f"A: {city_name}的天气是：{weather}")
  else:
    print("找不到函数")
else:
  print(f"A: {response_message['content']}")