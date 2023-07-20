import requests
import json

# 定义你要发送的数据
data = {
    "input": "我要画猫",
    "chatbot": [],
    "history": [],
    "userID": 123456
}

# 使用 requests 库发送 POST 请求
response = requests.post(
    "http://127.0.0.1:8000/gpt4_predict", 
    data=json.dumps(data),  # 将数据转换为 JSON 格式
    headers={'Content-Type':'application/json'}  # 设置请求头为 JSON 格式
)

# 输出服务器的响应
print(response.json())

""" 
{'chatbot': [['我要画猫', '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。']],
 'history': [{'role': 'user', 'content': '我要画猫'}, {'role': 'assistant', 'content': '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，
 比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。'}]}
"""