import gradio as gr
from utils import *

greetings = [("你好呀！", "您好！我是 ArtAgent ChatBot，一个与您交流艺术构思、生成和修改图像的AI助手。\n\n我调用了 GPT4，VisualGLM-6B 和 Stable Diffusion 模型。\n\n如果您想根据艺术讨论生成富有创意的图像，请点击 “Generate Creative Image” 按钮。如果您想严谨地修改图像，请点击 “Please Edit It!” 按钮。如果不满意，请重新生成。如果想开启一个新的创作主题，请点击 “Begin a New Topic” 按钮。\n\n我还在测试阶段，链路中也存在很多随机性：Sometimes a simple retry can make it better.")]

gr.Chatbot.postprocess = postprocess

with gr.Blocks(title="ArtAgent ChatBot") as demo:
    gr.HTML("""<h1 align="center"> 🎊 ArtAgent  ChatBot 🎊 </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(greetings).style(height=600)
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(container=False)
                    with gr.Column(scale=1, min_width=100):
                        submitBtn = gr.Button("Chat with ArtAgent 🚀",)
                        emptyBtn = gr.Button("Begin a New Topic",)
            with gr.Row():
                sd_width = gr.Slider(512, 1024, value=768, step=32, label="图片宽度", interactive=True)
                sd_height = gr.Slider(512, 1024, value=768, step=32, label="图片高度", interactive=True)
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Tab("Gallery"):
                    result_gallery = gr.Gallery(label='Output', show_label=False).style(preview=True)
                with gr.Tab("Upload Image"):
                    upload_image = gr.Image(label='Upload', brush_radius=30, show_label=True, interactive=True, type="pil", tool='color-sketch')
            with gr.Row():
                drawBtn = gr.Button("Generate Creative Image 🎨", variant="primary")
                editBtn = gr.Button("Please Edit It!", variant="primary")
            with gr.Tab("Sketchpad"):
                sketchpad = gr.Sketchpad(shape=(1000, 1000), brush_radius=5, type="pil", tool="color-sketch")

                # pil 调粗细，color-sketch 是画板颜色选取
                # brush_radius 是笔触粗细，gr.Sketchpad().style(height=280, width=280) 没用，只是扩大画板外框框的大小

    history = gr.State([])
    result_list = gr.State([])
    userID = gr.State(random.randint(0, 9999999))  # 用户在未刷新情况下随机给到一个id
    cnt = gr.State(0)

    def click_count():
        cnt = 0
        yield cnt

    submitBtn.click(gpt4_predict, [user_input, chatbot, history, userID], [chatbot, history], show_progress=True)  # 艺术讨论
    submitBtn.click(reset_user_input, [], [user_input])  # 发送完信息就清空。一次点击触发两个函数
    submitBtn.click(click_count, [], [cnt])  # 一次生图不满意，继续点击按钮，中间没说话：向他道歉

    editBtn.click(gpt4_sd_edit, [chatbot, history, result_list, userID, cnt, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    drawBtn.click(gpt4_sd_draw, [chatbot, history, result_list, userID, cnt, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    upload_image.change(read_image, [upload_image, chatbot, history, userID], [chatbot, history], show_progress=True)

    emptyBtn.click(reset_state, [chatbot, userID], [chatbot, history, cnt], show_progress=True)

os.makedirs('output', exist_ok=True)
demo.queue().launch(share=True, inbrowser=True, server_name='127.0.0.1', server_port=7006, favicon_path="./favicon.ico")