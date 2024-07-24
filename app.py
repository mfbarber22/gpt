import gradio as gr

# Import modules from other files
from chatbot import model_inference, EXAMPLES, chatbot
from voice_chat import respond

# Define Gradio theme
theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="violet",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('orbitron')]
)


# Create Gradio blocks for different functionalities

# Chat interface block
with gr.Blocks(
        css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}""",
) as chat:
    gr.Markdown("### Image Chat, Image Generation, Image classification and Normal Chat")
    gr.ChatInterface(
        fn=model_inference,
        chatbot = chatbot,
        examples=EXAMPLES,
        multimodal=True,
        cache_examples=False,
        autofocus=False,
        concurrency_limit=10,
    )

# Voice chat block
with gr.Blocks() as voice:
    gr.Markdown("Sometimes, it takes because of long queue")
    with gr.Row():
        audio_input = gr.Audio(label="Voice Chat (BETA)", sources="microphone", type="filepath", waveform_options=False)
        output = gr.Audio(label="OUTPUT", type="filepath", interactive=False, autoplay=True, elem_classes="audio")
    audio_input.change( fn=respond, inputs=[audio_input], outputs=[output], queue=False)

with gr.Blocks() as image:
    gr.HTML("<iframe src='https://kingnish-image-gen-pro.hf.space' width='100%' height='2000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as instant2:
    gr.HTML("<iframe src='https://kingnish-instant-video.hf.space' width='100%' height='3000px' style='border-radius: 8px;'></iframe>")

with gr.Blocks() as video:
    gr.Markdown("""More Models are coming""")
    gr.TabbedInterface([ instant2], ['Instant🎥'])     

# Main application block
with gr.Blocks(theme=theme, title="OpenGPT 4o DEMO") as demo:
    gr.Markdown("# OpenGPT 4o")
    gr.TabbedInterface([chat, voice, image, video], ['💬 SuperChat','🗣️ Voice Chat', '🖼️ Image Engine', '🎥 Video Engine'])

demo.queue(max_size=300)
demo.launch()