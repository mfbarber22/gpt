import gradio as gr
import spaces
from chatbot import model_inference, EXAMPLES, chatbot
from voice_chat import respond

# Define custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Roboto', sans-serif;
}

.main-header {
    text-align: center;
    color: #4a4a4a;
    margin-bottom: 2rem;
}

.tab-header {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.custom-chatbot {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.custom-button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.custom-button:hover {
    background-color: #2980b9;
}
"""

# Define Gradio theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Roboto'), "sans-serif"]
)

# Chat interface block
with gr.Blocks(css=custom_css) as chat:
    gr.Markdown("### 💬 OpenGPT 4o Chat", elem_classes="tab-header")
    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        examples=EXAMPLES,
        multimodal=True,
        cache_examples=False,
        autofocus=False,
        concurrency_limit=10
    )

# Voice chat block
with gr.Blocks() as voice:
    gr.Markdown("### 🗣️ Voice Chat", elem_classes="tab-header")
    gr.Markdown("Try Voice Chat from the link below:")
    gr.HTML('<a href="https://huggingface.co/spaces/KingNish/Voicee" target="_blank" class="custom-button">Open Voice Chat</a>')

# Image engine block
with gr.Blocks() as image:
    gr.Markdown("### 🖼️ Image Engine", elem_classes="tab-header")
    gr.Markdown("Image processing features coming soon!")

# Video engine block
with gr.Blocks() as video:
    gr.Markdown("### 🎥 Video Engine", elem_classes="tab-header")
    gr.Markdown("Video processing features coming soon!")

# Main application block
with gr.Blocks(theme=theme, title="OpenGPT 4o DEMO") as demo:
    gr.Markdown("# 🚀 OpenGPT 4o", elem_classes="main-header")
    gr.TabbedInterface(
        [chat, voice, image, video],
        ['💬 SuperChat', '🗣️ Voice Chat', '🖼️ Image Engine', '🎥 Video Engine']
    )

demo.queue(max_size=300)
demo.launch()