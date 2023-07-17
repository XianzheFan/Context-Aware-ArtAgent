import os
import requests, re, json, io, base64, os
import mdtex2html
from PIL import Image
import gradio as gr
from promptgen import *
import time
import random
import re
from io import BytesIO
from PIL import Image
import base64
from tqdm import tqdm
import cv2
import openai
import numpy as np


ART_ADVICE = "你是一个专业的艺术评论家。如果用户询问你的建议，你就根据之前的聊天记录，给用户一个绘画描述以提供灵感，以“您可以这样画这幅画”开头，要富有想象力，在150字以内，不要给出多种场景；如果用户提出自己的绘图建议，你要做出简要回答表示赞同。要使用中文回复，不要加双引号，不要说“我不具备生成图片的能力”"
UPLOAD_ADVICE = "你是一个专业的艺术评论家。给你关于用户图片的文字描述，你要先回复“收到图片”，接着另起一段，复述这段文字描述。然后另起一段，根据收到的文字描述，最好从增减或改变背景中的物体、变换绘画风格出发，提出专业有想象力的改进建议，不要有对比度、层次感这方面的建议。不要说“从你的描述中，您提到图片中”，而是要说“根据您上传的图片”这种类似的话。你要让用户认为图片是你自己理解的"
CN_TXT2IMG_PROMPT = "给你用户和艺术家的艺术讨论。分析该讨论的最终结果，从增减或改变背景中的物体、变换绘画风格出发，总结出艺术讨论后图片改进方向的几个关键词，开头加上原图的关键元素，作为文生图模型的英文prompt，不超过25词，不要有高对比度这种类似的词。回复时只写出英文prompt，一定不要加双引号和中文"
TXT2IMG_NEG_PROMPT = "给你用户和艺术家的艺术讨论。示例：用户不想画夜晚，你回复night scene；用户想画夜晚，你回复daytime。如果用户提出了想画的人、物、场景或风格，请把这些的反义词总结成英文关键词，不超过6个词。如果用户没有不想画的，就回复一个空格。一定不要加双引号，不要在开头写create或paint这种词。"
TXT2IMG_PROMPT = "给你用户和艺术家的艺术讨论，不要回复中文。若用户认为艺术家对图像描述不正确，你应该听从用户的要求。把用户选择的绘画主题放在开头，写出用于文生图模型的全英文prompt，来画一幅画，词数在50词以内。注意，如果描述比较长，需要提取主要意象和情景；如果较短，一定在突出绘画主体的基础上，运用想象力，添加一些内容以丰富细节。一定不要加双引号，不要在开头写create或paint这种词，直接描述画面。"
TRANLATE_IMAGE = "先说“图像生成完毕。”，然后另起一行，以“这幅画描绘了”开头，用中文写出这段英文描绘的场景，要优美流畅，不要让用户意识到你在翻译，而是认为你在点评一幅画。"


# glm_tokenizer = AutoTokenizer.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True)
glm_tokenizer = None
# glm_model = AutoModel.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True).half().quantize(4).cuda()
glm_model = None
# glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# glm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# glm_model = glm_model.eval()


def gpt4_predict(input, chatbot, history):
    """ input 是 gr.textbox(), history 是 List """
    chatbot.append((parse_text(input), ""))
    history.append(construct_user(str(input)))

    res = gpt4_api(ART_ADVICE, history)
    history.append(construct_assistant(res))
    print(history)

    chatbot[-1] = (parse_text(input), parse_text(res))
    yield chatbot, history


def construct_text(role, text):
    return {"role": role, "content": text}

def construct_user(text):
    return construct_text("user", text)

def construct_system(text):
    return construct_text("system", text)

def construct_assistant(text):
    return construct_text("assistant", text)


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


def reset_state(chatbot):
    chatbot.append((parse_text("我想开启一个新的绘画主题。"), parse_text("好的，您想创作什么主题的画呢？")))
    yield chatbot, [], 0


def clear_gallery():
    return [], []


def wrong_image(chatbot, history, cnt):
    if cnt > 0:
        chatbot.append((parse_text("这张图和我的想法不一致，请修改描述。"), parse_text("抱歉，我会重新修改描述，生成新的图像。")))
        history.append(construct_user("这张图和我的想法不一致，请修改描述。"))
        history.append(construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
    cnt = cnt + 1
    yield chatbot, history, cnt


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
    # 首先，确保numpy数组是uint8类型，且值在0-255范围内
    assert np_image.dtype == np.uint8
    assert np_image.min() >= 0
    assert np_image.max() <= 255
    
    # 将numpy数组转化为PIL图像
    img = Image.fromarray(np_image)
    
    # 将图像保存到指定路径
    img.save("output/edit-" + str(userID) + ".png")


def read_image(img, chatbot, history, userID):
    process_and_save_image(img, userID)
    chatbot.append((parse_text("请对这张图片给出建议。"), ""))

    response0 = call_visualglm_api(img, history)["result"]
    response = gpt4_api(UPLOAD_ADVICE, [construct_user(response0)])

    chatbot[-1] = (parse_text("请对这张图片给出建议。"), parse_text(response)) 

    history.append(construct_user("请对这张图片给出建议。"))
    history.append(construct_assistant(response))
    print(history)
    yield chatbot, history


def gpt4_sd_edit(chatbot, history, result_list, userID, cnt, step, cfg_scale, width, height):
    if cnt > 0:
        chatbot.append((parse_text("这张图和我的想法不一致，请修改描述。"), parse_text("抱歉，我会重新修改描述，生成新的图像。")))
        history.append(construct_user("这张图和我的想法不一致，请修改描述。"))
        history.append(construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
    cnt = cnt + 1

    image_path = "output/edit-" + str(userID) + ".png"
    pos_prompt = gpt4_api(CN_TXT2IMG_PROMPT, history)
    print(pos_prompt)
    new_images = controlnet_txt2img_api(image_path, pos_prompt, userID, step, cfg_scale, width, height, )
    
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


def controlnet_txt2img_api(image_path, pos_prompt, userID, step, cfg_scale, width, height, sampler="DPM++ 2M Karras", cn_module="canny", cn_model="control_v11p_sd15_canny [d14c016b]"):
    controlnet_image = Image.open(image_path)
    controlnet_image_data = encode_pil_to_base64(controlnet_image)
    txt2img_data = {
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "sampler_name": sampler,  # Euler也可
        "batch_size": 1,
        "step": step,
        "cfg_scale": cfg_scale,
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
                        "model": cn_model
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
        image.save('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(random.randint(1000, 9999)) + "-cn-"  + str(userID) + '.png')
    return image_list


def gpt4_sd_draw(chatbot, history, result_list, userID, cnt, step, cfg_scale, width, height):
    if cnt > 0:
        chatbot.append((parse_text("这张图和我的想法不一致，请修改描述。"), parse_text("抱歉，我会重新修改描述，生成新的图像。")))
        history.append(construct_user("这张图和我的想法不一致，请修改描述。"))
        history.append(construct_assistant("抱歉，我会重新修改描述，生成新的图像。"))
    cnt = cnt + 1

    image_path = "output/edit-" + str(userID) + ".png"
    pos_prompt = gpt4_api(TXT2IMG_PROMPT, history)
    print(f"pos_prompt: {pos_prompt}")
    neg_prompt = gpt4_api(TXT2IMG_NEG_PROMPT, history)
    print(f"neg_prompt: {neg_prompt}")
    new_images = call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, step, cfg_scale)
    
    new_image = new_images[0]
    new_image.save(os.path.join(image_path))  # 暂时存成edit.png
    result_list = [new_image] + result_list

    chatbot.append((parse_text("请基于之前的艺术讨论生成图片。"), ""))
    # response = gpt4_api(TRANLATE_IMAGE, pos_prompt)  # 不要直译了
    response = "图像生成完毕。\n\n" + call_visualglm_api(np.array(new_image))["result"]

    chatbot[-1] = (parse_text("请基于之前的艺术讨论生成图片。"), parse_text(response)) 
    history.append(construct_user("请基于之前的艺术讨论生成图片。"))
    history.append(construct_assistant(response))
    print(history)

    yield chatbot, history, result_list, new_images, cnt, result_list


def wrapper_gpt4_sd_draw(*args, **kwargs):
    # 用来处理生成器的函数
    generator = gpt4_sd_draw(*args, **kwargs)
    results = []
    for result in generator:
        results.append(result)
    return results


def translate_by_glm(word):
    for p in ["！", "，"]:
            word = word.replace(p, "。")
    words = word.split("。")
    trans_result = ""
    for word in words:
        word = word.strip()
        if len(word) > 0:
            trans_result += call_glm_api("翻译："+ word, [], 1024, 0.6, 0.9)["response"].strip()              
    return trans_result


def translate_by_youdao(word):
    trans_result = ''
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    key = {
        # 'type': "AUTO",
        'from': "zh-CHS",
        'to': "en",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url, data=key)
    if response.status_code == 200:
        list_trans = response.text
        result = json.loads(list_trans)
        for r in result['translateResult'][0]:
            trans_result += r['tgt']
    else:
        print(response.status_code)
        return word
    return trans_result


def call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, steps, cfg, user_input=""):
    url = "http://127.0.0.1:6016"
    payload = {
        "enable_hr": False,  # True画质更好但更慢
        "denoising_strength": 0.55,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "steps": steps,
        "negative_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs, multiple hands), (worst quality, low quality, extra digits, loli, loli face:1.2), " + neg_prompt + ", lowres, blurry, text, logo, artist name, watermark",
        "cfg_scale": cfg,
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
        image.save('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + "-" + str(userID) +'.png')

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


# 暂时不用
def call_glm_api(prompt, history, max_length, top_p, temperature):
    url = "http://127.0.0.1:8000"
    payload = {
        "prompt": prompt,
        "history": history,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature
    }
    response = requests.post(url, json=payload)
    response = response.json()
    # print(response)
    return response


# 暂时不用
def gen_image_description(user_input, chatbot, max_length, top_p, temperature, history, file_handle, from_api=True):
    # TODO 4.1
    def get_respond(prompt_history, prompt_input):
        if not from_api:
            response = ""
            for response_, history_ in glm_model.stream_chat(glm_tokenizer, prompt_input, prompt_history, max_length=max_length, top_p=top_p,
                                                temperature=temperature):
                response = response_
        else:
            response = call_glm_api(prompt_input, prompt_history, max_length, top_p, temperature)["response"]
        return response
    def write_log():
        file_handle.write("="*20 + "\n")
        file_handle.write("prompt_history:" + str(prompt_history) + "\n")
        file_handle.write("prompt_input:" + prompt_input + "\n")
        file_handle.write("response:" + response + "\n\n")
        file_handle.write("="*20+ "\n")

    # Step1 名词解释
    prompt_history = [["我接下来会给你一些名词，请你依次给出它们的解释。","好的，请给我一些指令。"]]
    prompt_input = str(f"名词解释：“{user_input}”，请详细解释这些词，并添加一些形象和内容以丰富细节，不要输出多余的信息")
    response = get_respond(prompt_history, prompt_input)
    print("Step1", response)
    write_log()

    # Step2 元素提取和总结
    prompt_history.append([prompt_input, response])
    prompt_input = str(f"请总结归纳你刚刚的解释，并为其添加一些视觉上的元素和细节，不要输出多余的信息。")
    response = get_respond(prompt_history, prompt_input)
    print("Step2", response)
    write_log()

    # Step3 作画素材
    prompt_history = [["我接下来会给你一些作画的指令，你只要回复出作画内容及对象，不需要你作画，不需要给我参考，请直接给出作画内容，不要输出不必要的内容，你只需回复作画内容。你听懂了吗", "听懂了。请给我一些作画的指令。"]]
    prompt_input = str(f"我现在要画一副画，这幅画关于：{response}。请帮我详细描述作画中的画面构图、画面主体和画面背景，并添加一些内容以丰富细节。回答中不要包含这一句话")
    response = get_respond(prompt_history, prompt_input)
    print("Step3", response)
    write_log()

    # 检测
    # retry_count = 0
    # check = get_respond([], str(f"这里有一段描述，{response}，这段描述是关于一个场景的吗？你仅需要回答“是”或“否”。"))
    # print("CHECK", check)
    # while ("不是" in check or "是" not in check) and retry_count < 3:
    #     response = get_respond(prompt_history, prompt_input)
    #     check = get_respond([], str(f"这里有一段描述，{response}，这段描述是关于一个场景、物体、动物或人物的吗？你仅需要回答“是”或“否”。"))
    #     retry_count += 1

    # if "不是" in check or "是" not in check:
    #     response = "抱歉, 我还不知道该怎么画，我可能需要更多学习。"
    #     chatbot.append((parse_text(user_input), parse_text(response)))
    #     history.append([chatbot[-1][0], chatbot[-1][1]])
    #     return chatbot, history, parse_text(response), "FAILED"

    chatbot.append((parse_text("请帮我画："+user_input), parse_text(response)))
    history.append([chatbot[-1][0], chatbot[-1][1]])

    # Step4 作画素材
    prompt_history = [["下面我将给你一段话，请你帮我抽取其中的图像元素，忽略其他非图像的描述，将抽取结果以逗号分隔，一定不要输出多余的内容和符号","听懂了，请给我一段文字。"]]
    prompt_input = str(f"以下是一段描述，抽取其中包括{TAG_STRING}的图像元素，忽略其他非图像的描述，将抽取结果以逗号分隔：{response}。 {user_input}")
    response = get_respond(prompt_history, prompt_input)
    print("Step4", response)
    write_log()

    # print(history[-1])
    return chatbot, history, parse_text(response) + "\n" + chatbot[-1][1], "SUCCESS"


# 暂时不用
def sd_predict(user_input, chatbot, max_length, top_p, temperature, history, width, height, steps, cfg, result_list, userID):
    file_handle = open('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + '.txt', 'w', encoding="utf8")
    
    # Step 1 use ChatGLM-6B associate image description
    # !!! Currently, we don't take history into consideration
    chatbot, history, image_description, code = gen_image_description(user_input, chatbot, max_length, top_p, temperature, history, file_handle)

    if code != "SUCCESS":
        yield chatbot, history, result_list, result_list
    else:
        # image_description = history[-1][1]
        # image_description = str("").join(image_description.split('\n')[1:])
        # stop_words = ["好的", "我", "将", "会", "画作", "关于", "一张", "画"]
        stop_words = ["\t", "\r", '-', '*', '·', "<br>"]
        for word in stop_words:
            image_description = image_description.replace(word, "\n")
        image_description += "\n"
        # print(image_description)
        tag_dict = {}

        # base on re        
        # for tag_class in TAG_CLASSES + ["构图", "主体", "背景", "内容"]:
        #     pat = r'{}：.*[\n]'.format(tag_class)
        #     # print(pat)
        #     pat = re.compile( pat)
        #     find = pat.findall(image_description)
        #     if len(find) > 0:
        #         if "不清楚" not in find[0] and "无" not in find[0] and "没有描述" not in find[0] and "不知道" not in find[0] and len(find[0]) > 1:
        #             tag_dict[tag_class] = find[0][len(tag_class) + 1: -1]
        
        # base on find
        TAG_CLASSES_ = TAG_CLASSES + SIMI_TAG_CLASSES
        tag_pos_dict = {}
        for t in TAG_CLASSES_:
            pos = image_description.find(t+"：")
            if pos != -1:
                tag_pos_dict[t] = pos
        tag_pos_dict = sorted(tag_pos_dict.items(), key = lambda kv:(kv[1], kv[0]))
        tag_pos_dict = [(index, a[0], a[1]) for index, a in enumerate(tag_pos_dict)] + [(len(tag_pos_dict), "", len(image_description))]
        print(tag_pos_dict)
        for index in range(len(tag_pos_dict) - 1):
            l = tag_pos_dict[index][2] + len(tag_pos_dict[index][1]) + 1
            r = tag_pos_dict[index+1][2]
            tmp = image_description[l:r]
            if "不清楚" not in tmp and "无" not in tmp and "没有描述" not in tmp and "不知道" not in tmp and "未指定" not in tmp:
                tmp = tmp.replace('\n', "")
                tag_dict[tag_pos_dict[index][1]] = tmp

        print(tag_dict)
        file_handle.write(str(tag_dict) + "\n")
        
        if len(tag_dict) <= 1:
            for word in TAG_CLASSES + ["\n", "\t", "\r", "<br>"] + ["根据描述无法识别", "无", "没有描述", "不知道", "不清楚"]:
                image_description = image_description.replace(word, ", ")
            tag_dict["其他"] = image_description
            print(tag_dict)
        
        tag_dict = dict([(tag, translate_by_youdao(tag_dict[tag]).lower() if tag in TAG_CLASSES else translate_by_youdao(tag_dict[tag]).lower()) for tag in tag_dict if len(tag_dict[tag]) > 0])
        print(tag_dict)        
        file_handle.write(str(tag_dict) + "\n")
        # image_description = translate(image_description)
        # print(image_description)

        # Step 2 use promprGenerater get Prompts
        # prompt_list = gen_prompts(image_description, batch_size=4)
        # print(prompt_list)
        # yield chatbot, history, result_list, []

        # Alternative plan
        # prompt_list = [ enhance_prompts(image_description) ] * 4
        prompt_list = tag_extract(tag_dict)
        print(prompt_list[0])
        file_handle.write("\n".join(["Prompt:"+p[0]+"\nNegative Prompt:"+p[1] for p in prompt_list]))

        # Show Prompts
        prompt_text = "\n\n Prompt:\n\n " + str(prompt_list[0][0]) + "\n\nNegative Prompt: \n\n" + str(prompt_list[0][1])
        chatbot[-1] = (chatbot[-1][0], chatbot[-1][1] + prompt_text)


        file_handle.close()
        prompt_list = [prompt_list[0]]
        # Step 3 use SD get images
        for pos_prompt, neg_prompt in prompt_list:
            new_images = call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, steps, cfg, user_input)
            result_list = result_list + new_images
            yield chatbot, history, result_list, new_images
        yield chatbot, history, result_list, result_list