import re
import gradio as gr
from huggingface_hub import InferenceClient

client2 = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions2 = "[SYSTEM] You are the Best AI, you can solve complex problems you answer in short , simple and easy language.[USER]"

def text(prompt):
    generate_kwargs = dict(
        temperature=0.5,
        max_new_tokens=5,
        top_p=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions2 + prompt + "[BOT]"
    stream = client2.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output  


client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions = "[SYSTEM] You will be provided with text, and your task is to classify task tasks are (text generation, image generation, tts) answer with only task type that prompt user give, do not say anything else and stop as soon as possible. Example: User- What is friction , BOT - text generation [USER]"

def classify_task(prompt):
    generate_kwargs = dict(
        temperature=0.5,
        max_new_tokens=5,
        top_p=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions + prompt + "[BOT]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    if 'text' in output.lower():
        user = text(prompt)
    elif 'image' in output.lower():
        return 'Image Generation'
    else:
        return 'Unknown Task'




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
