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
iface = gr.Interface(
    fn=classify_task,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs='text',
    title='AI Task Classifier Chatbot',
    description='This chatbot classifies your prompt into different AI tasks.'
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
