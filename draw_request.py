import requests
import json
from utils import ImageRequest, HistoryItem

history = [
    {'role': 'user', 'content': '我要画猫'}, 
    {'role': 'assistant', 'content': '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。'}
]
history_data = [HistoryItem.parse_obj(item).dict() for item in history]

url = 'http://localhost:8000/gpt4_sd_draw'
data = ImageRequest(
    chatbot = [['我要画猫', '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。']],
    history = history_data,
    userID=12345,
    cnt=0,
    width=768,
    height=768
)
response = requests.post(url, json=data.dict())
# http://localhost:8000/static/images/3eabaa98-0e94-43a8-88a3-000371d9e5fb.png