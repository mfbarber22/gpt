import gradio as gr

# Import modules from other files
from chatbot import chatbot, model_inference, EXAMPLES
from live_chat import videochat

# Define Gradio theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif']
).set(
    body_background_fill_dark="#111111",
    block_background_fill_dark="#111111",
    block_border_width="1px",
    block_title_background_fill_dark="#1e1c26",
    input_background_fill_dark="#292733",
    button_secondary_background_fill_dark="#24212b",
    border_color_primary_dark="#343140",
    background_fill_secondary_dark="#111111",
    color_accent_soft_dark="transparent"
)


with gr.Blocks() as voice:    
    gr.Markdown("## Temproraly Not Working (Update in Progress)")

# Chat interface block
with gr.Blocks(
        fill_height=True,
        css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
) as chat:
    gr.Markdown("### Chat with Image, Chat with Video, Image Generation and Normal Chat")
    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        examples=EXAMPLES,
        multimodal=True,
        cache_examples=False,
        additional_inputs=[
            gr.Checkbox(label="Web Search", value=False),
        ],
    )    

# Live chat block
with gr.Blocks() as livechat:
    gr.Interface(
        fn=videochat,
        inputs=[gr.Image(type="pil",sources="webcam", label="Upload Image"), gr.Textbox(label="Prompt", value="what he is doing")],
        outputs=gr.Textbox(label="Answer")
    )

with gr.Blocks() as image:
    gr.HTML("<iframe src='https://kingnish-image-gen-pro.hf.space' width='100%' height='2000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as instant2:
    gr.HTML("<iframe src='https://kingnish-instant-video.hf.space' width='100%' height='3000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as video:
    gr.Markdown("""More Models are coming""")
    gr.TabbedInterface([ instant2], ['Instant🎥'])   

# Main application block
with gr.Blocks(theme=theme, title="OpenGPT 4o DEMO") as demo:
    gr.Markdown("# OpenGPT 4o\n### Try its small,fast and unlimited version from here: [OpenGPT 4o Mini](https://huggingface.co/spaces/KingNish/OpenGPT-4o-mini)")
    gr.TabbedInterface([chat, voice, livechat, image, video], ['💬 SuperChat','🗣️ Voice Chat','📸 Live Chat', '🖼️ Image Engine', '🎥 Video Engine'])

demo.queue(max_size=300)
demo.launch()