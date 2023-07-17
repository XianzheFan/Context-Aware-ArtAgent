import gradio as gr
from utils import *

greetings = [("你好呀！", "您好！我是 ArtAgent ChatBot，一个与您交流艺术构思、生成和修改图像的AI助手。\n\n我调用了 GPT4，VisualGLM-6B 和 Stable Diffusion 模型。\n\n如果您想根据艺术讨论生成富有创意的图像，请点击 “Generate Creative Image” 按钮。如果您想严谨地修改图像，请点击 “Please Edit It!” 按钮。如果不满意，请重新生成。如果想开启一个新的创作主题，请点击 “Begin a New Topic” 按钮。\n\n我还在测试阶段，链路中也存在很多随机性：Sometimes a simple retry can make it better.")]

gr.Chatbot.postprocess = postprocess

with gr.Blocks(title="ArtAgent ChatBot") as demo:
    gr.HTML("""<h1 align="center"> 🎊 ArtAgent  ChatBot 🎊 </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(greetings).style(height=640)
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(container=False)
                    with gr.Column(scale=1, min_width=100):
                        submitBtn = gr.Button("Chat with ArtAgent 🚀",)
                        emptyBtn = gr.Button("Begin a New Topic",)
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Tab("Gallery"):
                    result_gallery = gr.Gallery(label='Output', show_label=False).style(preview=True)
                with gr.Tab("Upload Image"):
                    upload_image = gr.Image(label='Upload', show_label=True)
                with gr.Tab("Sketchpad"):
                    sketchpad = gr.Sketchpad()
            with gr.Row():
                drawBtn = gr.Button("Generate Creative Image 🎨", variant="primary")
                editBtn = gr.Button("Please Edit It!", variant="primary")
            with gr.Row():
                with gr.Tab("Settings"):
                    with gr.Tab(label="Stable Diffusion"):
                        with gr.Column(min_width=100):
                            # clearBtn = gr.Button("Clear Gallery")
                            with gr.Row():
                                sd_width = gr.Slider(512, 1024, value=768, step=32, label="图片宽度", interactive=True)
                                sd_height = gr.Slider(512, 1024, value=768, step=32, label="图片高度", interactive=True)
                            with gr.Row():
                                sd_steps = gr.Slider(8, 40, value=32, step=4, label="生成图像迭代次数", interactive=True)
                                sd_cfg = gr.Slider(4, 20, value=7, step=0.5, label="提示词相关性", interactive=True)
                    with gr.Tab(label="ChatGLM-6B"):
                        with gr.Column(min_width=100):
                            max_length = gr.Slider(0, 4096, value=2048, step=64.0, label="对话长度限制", interactive=True)
                            with gr.Row():
                                top_p = gr.Slider(0, 1, value=0.6, step=0.01, label="Top P", interactive=True)
                                temperature = gr.Slider(0, 1, value=0.90, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    result_list = gr.State([])
    userID = gr.State(0)  # 用户在未刷新情况下随机给到一个id
    cnt = gr.State(0)
    def addID():
        userID = random.randint(0, 99999)
        yield userID
    def click_count():
        cnt = 0
        yield cnt

    submitBtn.click(gpt4_predict, [user_input, chatbot, history], [chatbot, history], show_progress=True)  # 艺术讨论
    submitBtn.click(reset_user_input, [], [user_input])  # 发送完信息就清空。一次点击触发两个函数
    submitBtn.click(click_count, [], [cnt])  # 一次生图不满意，继续点击按钮，中间没说话：向他道歉

    editBtn.click(gpt4_sd_edit, [chatbot, history, result_list, userID, cnt, sd_steps, sd_cfg, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    drawBtn.click(gpt4_sd_draw, [chatbot, history, result_list, userID, cnt, sd_steps, sd_cfg, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    upload_image.change(read_image, [upload_image, chatbot, history, userID], [chatbot, history], show_progress=True)

    emptyBtn.click(reset_state, [chatbot], [chatbot, history, cnt], show_progress=True)

os.makedirs('output', exist_ok=True)
demo.queue().launch(share=True, inbrowser=True, server_name='127.0.0.1', server_port=6006, favicon_path="./favicon.ico")