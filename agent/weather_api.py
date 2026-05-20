import requests

key = "7815c66716eeb680e1227c195fa9aec8"

mapping = {
    "北京": "110000",
    "上海": "110101",
    "广州": "440100",
    "深圳": "440300",
    "杭州": "330100",
    "南京": "320100",
    "西安": "610100",
    "武汉": "420100",
    "成都": "510100",
    "天津": "120100",
    "重庆": "500100",
    "苏州": "320500",
    "无锡": "320400",
    "厦门": "350200",
    "郑州": "410100",
}

def get_weather_api (location):
    location_code = mapping.get(location)
    weather_resp = requests.get(f"https://restapi.amap.com/v3/weather/weatherInfo?city={location_code}&key={key}")
    weather_json = weather_resp.json()
    return weather_json["lives"][0]["weather"]
