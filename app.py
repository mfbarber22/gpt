import re
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions = "You will be provided with text, and your task is to classify task tasks are (text generation, image generation, pdf chat, image text to text, image classification, summarization, translation , tts) answer with only task do not say anything else and stop as soon as possible."


def classify_task(prompt):
    generate_kwargs = dict(
        temperature=0.5,
        max_new_tokens=5,
        top_p=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions + prompt
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output       

# Create the Gradio interface
with gr.Blocks() as demo:
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
