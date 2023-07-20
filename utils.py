# -*- coding: utf-8 -*-
import os
import requests, re, json, io, base64, os
import mdtex2html
from PIL import Image
import gradio as gr
from promptgen import *
import time
import random
import re
from PIL import Image
import base64
import cv2
import openai
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import uuid


ART_ADVICE = "你是一个专业的艺术评论家。如果用户询问你的建议，你就根据之前的聊天记录，给用户一个绘画描述以提供灵感，以“您可以这样画这幅画”开头，要富有想象力，在150字以内，不要给出多种场景；如果用户提出自己的绘图建议，你要做出简要回答表示赞同。要使用中文回复，不要加双引号，不要说“我不具备生成图片的能力”"
UPLOAD_ADVICE = "你是一个专业的艺术评论家。给你关于用户图片的文字描述，你要先回复“收到图片”，接着另起一段，复述这段文字描述。然后另起一段，根据收到的文字描述，最好从增减或改变背景中的物体、变换绘画风格出发，提出专业有想象力的改进建议，不要有对比度、层次感这方面的建议。不要说“从你的描述中，您提到图片中”，而是要说“根据您上传的图片”这种类似的话。你要让用户认为图片是你自己理解的"
CN_TXT2IMG_PROMPT = "给你用户和艺术家的艺术讨论。分析该讨论的最终结果，从增减或改变背景中的物体、变换绘画风格出发，总结出艺术讨论后图片改进方向的几个关键词，开头加上原图的关键元素，作为文生图模型的英文prompt，不超过25词，不要有高对比度这种类似的词。回复时只写出英文prompt，一定不要加双引号和中文"
TXT2IMG_NEG_PROMPT = "给你用户和艺术家的艺术讨论。示例：用户不想画夜晚，你回复night scene；用户想画夜晚，你回复daytime。如果用户提出了想画的人、物、场景或风格，请把这些的反义词总结成全英文关键词，不超过6个词。如果用户没有不想画的，就回复一个空格。一定不要加双引号，不要在开头写create或paint这种词。不要使用中文"
TXT2IMG_PROMPT = "给你用户和艺术家的艺术讨论，不要回复中文。若用户认为艺术家对图像描述不正确，你应该听从用户的要求。把用户选择的绘画主题放在开头，写出用于文生图模型的全英文prompt，来画一幅画，词数在50词以内。注意，如果描述比较长，需要提取主要意象和情景；如果较短，一定在突出绘画主体的基础上，运用想象力，添加一些内容以丰富细节。一定不要加双引号，不要在开头写create或paint这种词，直接描述画面。"
TRANLATE_IMAGE = "先说“图像生成完毕。”，然后另起一行，以“这幅画描绘了”开头，用中文写出这段英文描绘的场景，要优美流畅，不要让用户意识到你在翻译，而是认为你在点评一幅画。"


def write_json(userID, output):
    with open('output/' + str(userID) + '.json', 'a', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)  # False，可保存utf-8编码
        f.write('\n')


class ChatbotData(BaseModel):
    input: str
    chatbot: List[str]
    history: List[str]
    userID: int

class HistoryItem(BaseModel):
    role: str
    content: str

class ImageRequest(BaseModel):
    chatbot: List[List[str]]
    history: List[Dict[str, str]]
    # result_list: List[str]
    userID: int
    cnt: int
    width: int
    height: int

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# 可以通过 URL /static/image.png 来访问文件
@app.post("/gpt4_predict")
def gpt4_predict(data: ChatbotData):
    """ input 是 gr.textbox(), history 是 List """
    data.chatbot.append((parse_text(data.input), ""))
    user_output = construct_user(str(data.input))
    data.history.append(user_output)

    res = gpt4_api(ART_ADVICE, data.history)
    assistant_output = construct_assistant(res)
    data.history.append(assistant_output)

    write_json(data.userID, user_output)
    write_json(data.userID, assistant_output)
    print(data.history)
    data.chatbot[-1] = (parse_text(data.input), parse_text(res))
    
    return {"chatbot": data.chatbot, "history": data.history}

# uvicorn utils:app --reload
# uvicorn main:app --reload --port 8080  默认是8000端口，可以改成别的
# http://127.0.0.1:8000/docs 是api文档


@app.post("/gpt4_sd_draw")
def gpt4_sd_draw(data: ImageRequest, request: Request):
    image_path = "output/edit-" + str(data.userID) + ".png"
    pos_prompt = gpt4_api(TXT2IMG_PROMPT, data.history)
    print(f"pos_prompt: {pos_prompt}")
    neg_prompt = gpt4_api(TXT2IMG_NEG_PROMPT, data.history)
    print(f"neg_prompt: {neg_prompt}")
    write_json(data.userID, construct_prompt(pos_prompt + "\n" + neg_prompt))
    new_images = call_sd_t2i(data.userID, pos_prompt, neg_prompt, data.width, data.height)
    
    new_image = new_images[0]
    new_image.save(os.path.join(image_path))  # 暂时存成edit.png
    static_path = "static/images/" + str(uuid.uuid4()) + ".png"
    print("图片链接 http://localhost:8000/" + static_path)
    new_image.save(static_path)
    # 构造URL
    image_url = request.url_for("static", path=static_path)

    if data.cnt > 0:
        data.chatbot.append((parse_text("这张图和我的想法不一致，请修改描述。"), parse_text("抱歉，我会重新修改描述，生成新的图像。")))
        data.history.append(construct_user("这张图和我的想法不一致，请修改描述。"))
        data.history.append(construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
        write_json(data.userID, construct_user("这张图和我的想法不一致，请修改描述。"))
        write_json(data.userID, construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
    data.cnt = data.cnt + 1

    data.chatbot.append((parse_text("请基于之前的艺术讨论生成图片。"), ""))
    response = "图像生成完毕。\n\n" + call_visualglm_api(np.array(new_image))["result"]

    data.chatbot[-1] = (parse_text("请基于之前的艺术讨论生成图片。"), parse_text(response)) 
    data.history.append(construct_user("请基于之前的艺术讨论生成图片。"))
    data.history.append(construct_assistant(response))
    write_json(data.userID, construct_user("请基于之前的艺术讨论生成图片。"))
    write_json(data.userID, construct_assistant(response))
    print(data.history)

    return {"chatbot": data.chatbot, "history": data.history,
            "image_url": str(image_url), "userID": data.userID,
            "cnt": data.cnt, "width": data.width, "height": data.height}


def construct_text(role, text):
    return {"role": role, "content": text}

def construct_user(text):
    return construct_text("user", text)

def construct_system(text):
    return construct_text("system", text)

def construct_assistant(text):
    return construct_text("assistant", text)

def construct_prompt(text):
    return construct_text("prompt", text)

def construct_photo(text):
    return construct_text("photo", text)


def gpt4_api(system, history):
    """ 返回str，参数为str,List """
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=[construct_system(system), *history])
        return response['choices'][0]['message']['content']
    except openai.error.ServiceUnavailableError:
        print('The server is overloaded or not ready yet. Please try again later.')
        return None
    except Exception as e:
        print(f'Unexpected error occurred: {e}')
        return None


def reset_user_input():
    return gr.update(value='')


def reset_state(chatbot, userID):
    chatbot.append((parse_text("我想开启一个新的绘画主题。"), parse_text("好的，您想创作什么主题的画呢？")))
    write_json(userID, construct_user("我想开启一个新的绘画主题。"))
    write_json(userID, construct_assistant("好的，您想创作什么主题的画呢？"))
    yield chatbot, [], 0


def clear_gallery():
    return [], []


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):  # 便于文本以html形式显示
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def process_and_save_image(np_image, userID):
    # 如果输入图像不是numpy数组，则进行转换
    if not isinstance(np_image, np.ndarray):
        np_image = np.array(np_image)
        
    # 确保我们有一个有效的numpy数组
    if np_image is None:
        raise ValueError("Image processing failed and resulted in None.")
    
    # 如果numpy数组不是uint8类型，则进行转换
    if np_image.dtype != np.uint8:
        np_image = np_image.astype(np.uint8)
        
    # 首先，确保numpy数组是uint8类型，且值在0-255范围内
    assert np_image.dtype == np.uint8
    assert np_image.min() >= 0
    assert np_image.max() <= 255
    
    # 将numpy数组转化为PIL图像
    img = Image.fromarray(np_image)
    
    # 将图像保存到指定路径
    img_path = 'output/' + time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(random.randint(1000, 9999)) + "-upload-"  + str(userID) + '.png'
    write_json(userID, construct_photo(img_path))
    img.save(img_path)
    img.save("output/edit-" + str(userID) + ".png")


def read_image(img, chatbot, history, userID):
    # 如果输入图像是PIL图像，将其转换为numpy数组
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    process_and_save_image(img, userID)
    chatbot.append((parse_text("请对这张图片给出建议。"), ""))

    response0 = call_visualglm_api(img, history)["result"]
    response = gpt4_api(UPLOAD_ADVICE, [construct_user(response0)])

    chatbot[-1] = (parse_text("请对这张图片给出建议。"), parse_text(response)) 

    history.append(construct_user("请对这张图片给出建议。"))
    history.append(construct_assistant(response))
    write_json(userID, construct_user("请对这张图片给出建议。"))
    write_json(userID, construct_assistant(response))
    print(history)
    yield chatbot, history


def gpt4_sd_edit(chatbot, history, result_list, userID, cnt, width, height):
    if cnt > 0:
        chatbot.append((parse_text("这张图和我的想法不一致，请修改描述。"), parse_text("抱歉，我会重新修改描述，生成新的图像。")))
        history.append(construct_user("这张图和我的想法不一致，请修改描述。"))
        history.append(construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
        write_json(userID, construct_user("这张图和我的想法不一致，请修改描述。"))
        write_json(userID, construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
    cnt = cnt + 1

    image_path = "output/edit-" + str(userID) + ".png"
    pos_prompt = gpt4_api(CN_TXT2IMG_PROMPT, history)
    print(pos_prompt)
    write_json(userID, construct_prompt(pos_prompt))
    new_images = controlnet_txt2img_api(image_path, pos_prompt, userID, width, height)
    
    # 仅保存第一张图片
    if new_images:  # 检查列表是否为空
        new_image = new_images[0]
        new_image.save(os.path.join(image_path))  # 暂时存成edit.png
        result_list = [new_image] + result_list  # 只添加第一张图片到结果列表

    yield chatbot, history, result_list, new_images, cnt, result_list  # 永远是新生成的图放在gallery第一位


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data).decode("utf-8")


def controlnet_txt2img_api(image_path, pos_prompt, userID, width, height, sampler="DPM++ 2M Karras", cn_module="canny", cn_model="control_v11p_sd15_canny [d14c016b]"):
    controlnet_image = Image.open(image_path)
    controlnet_image_data = encode_pil_to_base64(controlnet_image)
    txt2img_data = {
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "sampler_name": sampler,  # Euler也可
        "batch_size": 1,
        "step": 32,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "enabled": True,
        "negtive_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs), (worst quality, low quality, extra digits, loli, loli face:1.2), lowres, blurry, text, logo, artist name, watermark",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": controlnet_image_data,
                        "module": cn_module,
                        "model": cn_model,
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }

    response = requests.post(url=f'http://127.0.0.1:6016/sdapi/v1/txt2img', json=txt2img_data)
    print(txt2img_data["width"])
    print(txt2img_data["height"])
    r = response.json()
    image_list = []
    os.makedirs('output', exist_ok=True)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        output_path = 'output/' + time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(random.randint(1000, 9999)) + "-cn-"  + str(userID) + '.png'
        image.save(output_path)
        write_json(userID, construct_photo(output_path))
    return image_list


def call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, user_input=""):
    url = "http://127.0.0.1:6016"
    payload = {
        "enable_hr": True,  # True画质更好但更慢
        # "enable_hr": False,  # True画质更好但更慢
        "denoising_strength": 0.55,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "steps": 32,
        "negative_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs, multiple hands), (worst quality, low quality, extra digits, loli, loli face:1.2), " + neg_prompt + ", lowres, blurry, text, logo, artist name, watermark",
        "cfg_scale": 7,
        "batch_size": 1,
        "n_iter": 1,
        "width": width,
        "height": height,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    print(payload["width"])
    print(payload["height"])
    r = response.json()
    image_list = []
    os.makedirs('output', exist_ok=True)
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        output_path = 'output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + "-" + str(userID) +'.png'
        image.save(output_path)
        write_json(userID, construct_photo(output_path))

    return image_list


def call_visualglm_api(img, history=[]):
    history = []  # 先不给历史
    prompt="详细描述这张图片，包括画中的人、景、物、构图、颜色等"
    url = "http://127.0.0.1:8080"

    # 将BGR图像转换为RGB图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_byte = cv2.imencode('.png', img)[1]
    img_base64 = base64.b64encode(img_byte).decode("utf-8")
    payload = {
        "image": img_base64,
        "text": prompt,
        "history": history
    }
    response = requests.post(url, json=payload)
    response = response.json()
    return response