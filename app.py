import re
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions = """<s> [INST] You will be provided with text, and your task is to classify task tasks are (text generation, image generation, pdf chat, image text to text, image classification, summarization, translation , tts) """


def classify_task(prompt):
    generate_kwargs = dict(
        temperature=0.5,
        max_new_tokens=1024,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions + prompt + "[/INST]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text

# Define the classification function
def classify_task2(prompt):
    # Here you would implement the logic to classify the prompt
    # For example, using if-elif-else statements or a machine learning model
    if 'generate text' in prompt.lower():
        return 'Text Generation'
    elif 'generate image' in prompt.lower():
        return 'Image Generation'
    elif 'pdf chat' in prompt.lower():
        return 'PDF Chat'
    elif 'image to text' in prompt.lower():
        return 'Image Text to Text'
    elif 'classify image' in prompt.lower():
        return 'Image Classification'
    else:
        return 'Unknown Task'

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""
<center><h1>Emoji Translator 🤗😻</h1>
<h3>Translate any text into emojis, and vice versa!</h3>
</center>
""")

    gr.Markdown("""
# Text to Emoji 📖➡️😻
""")
    with gr.Row():
        text_uesr_input = gr.Textbox(label="Enter text 📚")
        output = gr.Textbox(label="Translation")
    with gr.Row():
        translate_btn = gr.Button("Translate 🚀")
        translate_btn.click(fn=classify_task, inputs=text_uesr_input,
                            outputs=output, api_name="translate_text")

# Launch the app
if __name__ == "__main__":
    demo.launch()
