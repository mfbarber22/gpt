import gradio as gr
from chatbot import model_inference, EXAMPLES, chatbot

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

# Main application block with only the Super Chat tab
with gr.Blocks(theme=theme, title="OpenGPT 4o DEMO", css=custom_css) as demo:
    gr.Markdown("# 🚀 OpenGPT 4o", elem_classes="main-header")
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

demo.queue(max_size=300)
demo.launch()
