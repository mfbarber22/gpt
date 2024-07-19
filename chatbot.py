import os
import time
import copy
import requests
import random
from threading import Thread
from typing import List, Dict, Union
import subprocess
# Install flash attention, skipping CUDA build if necessary
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)
import torch
import gradio as gr
from bs4 import BeautifulSoup
import datasets
from transformers import LlavaProcessor, LlavaForConditionalGeneration, TextIteratorStreamer
from huggingface_hub import InferenceClient
from PIL import Image
import spaces
from functools import lru_cache
import cv2
import re
import io  # Add this import for working with image bytes

# You can also use models that are commented below
# model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
model_id = "llava-hf/llava-interleave-qwen-7b-hf"
# model_id = "llava-hf/llava-interleave-qwen-7b-dpo-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, use_flash_attention_2=True, low_cpu_mem_usage=True)
model.to("cuda")
# Credit to merve for code of llava interleave qwen

def sample_frames(video_file) :
    try:
        video = cv2.VideoCapture(video_file)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = 12
        interval = total_frames // num_frames
        frames = []
        for i in range(total_frames):
            ret, frame = video.read()
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not ret:
                continue
            if i % interval == 0:
                frames.append(pil_img)
        video.release()
        return frames
    except:
        frames=[]
        return frames
        

# Path to example images
examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "Bitcoin price live",
        }
    ],
    [
        {
            "text": "Today News about AI",
        }
    ],
    [
        {
            "text": "What unusual happens in this video.",
            "files": [f"{examples_path}/example_video/accident.gif"],
        }
    ],
    [
        {
            "text": "What's name of superhero in this clip",
            "files": [f"{examples_path}/example_video/spiderman.gif"],
        }
    ],
    [
        {
            "text": "What's written on this paper",
            "files": [f"{examples_path}/example_images/paper_with_text.png"],
        }
    ],
    [
        {
            "text": "Who are they? Tell me about both of them",
            "files": [f"{examples_path}/example_images/elon_smoking.jpg",
                      f"{examples_path}/example_images/steve_jobs.jpg", ]
        }
    ],
    [
        {
            "text": "Create five images of supercars, each in a different color.",
        }
    ],
    [
        {
            "text": "Create a Photorealistic image of the Eiffel Tower.",
        }
    ],
    [
        {
            "text": "Create an ad script for this product.",
            "files": [f"{examples_path}/example_images/shampoo.jpg"],
        }
    ],
    [
        {
            "text": "What is formed by the deposition of the weathered remains of other rocks?",
            "files": [f"{examples_path}/example_images/ai2d_example.jpeg"],
        }
    ],
    [
        {
            "text": "What's unusual about this image?",
            "files": [f"{examples_path}/example_images/dragons_playing.png"],
        }
    ],
]

# Set bot avatar image
BOT_AVATAR = "OpenAI_logo.png"

# Perform a Google search and return the results
@lru_cache(maxsize=128) 
def extract_text_from_webpage(html_content):
    """Extracts visible text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
        tag.extract()
    visible_text = soup.get_text(strip=True)
    return visible_text

# Perform a Google search and return the results
def search(term, num_results=3, lang="en", advanced=True, timeout=5, safe="active", ssl_verify=None):
    """Performs a Google search and returns the results."""
    start = 0
    all_results = []
    # Limit the number of characters from each webpage to stay under the token limit
    max_chars_per_page = 8000  # Adjust this value based on your token limit and average webpage length
    
    with requests.Session() as session: 
        resp = session.get(  
            url="https://www.google.com/search",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}, 
                params={
                    "q": term,
                    "num": num_results,
                    "udm": 14,
                },
                timeout=timeout,
                verify=ssl_verify,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        result_block = soup.find_all("div", attrs={"class": "g"})
        for result in result_block:
            link = result.find("a", href=True)
            if link:
                link = link["href"]
                try:
                    webpage = session.get(link, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0"}) 
                    webpage.raise_for_status()
                    visible_text = extract_text_from_webpage(webpage.text)
                        # Truncate text if it's too long
                    if len(visible_text) > max_chars_per_page:
                        visible_text = visible_text[:max_chars_per_page]
                    all_results.append({"link": link, "text": visible_text})
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching or processing {link}: {e}")
                    all_results.append({"link": link, "text": None})
            else:
                all_results.append({"link": None, "text": None}) 
    return all_results

# Format the prompt for the language model
def format_prompt(user_prompt, chat_history):
    prompt = "<s>"
    for item in chat_history:
        # Check if the item is a tuple (text response)
        if isinstance(item, tuple):
            prompt += f"[INST] {item[0]} [/INST]"  # User prompt
            prompt += f" {item[1]}</s> "           # Bot response
        # Otherwise, assume it's related to an image - you might need to adjust this logic
        else:
            # Handle image representation in the prompt, e.g., add a placeholder
            prompt += f" [Image] " 
    prompt += f"[INST] {user_prompt} [/INST]"
    return prompt


client_mixtral = InferenceClient("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
client_mistral = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
generate_kwargs = dict( max_new_tokens=4000, do_sample=True, stream=True, details=True, return_full_text=False )

@spaces.GPU(duration=60, queue=False)
def model_inference( user_prompt, chat_history, web_search):
    if not user_prompt["files"]:
        if web_search is True:
            
            gr.Info("Searching Web")
            
            web_results = search(user_prompt["text"])
            
            gr.Info("Extracting relevant Info")
            
            web2 = ' '.join([f"Link: {res['link']}\nText: {res['text']}\n\n" for res in web_results])
            
            messages = f"<|im_start|>system\nYou are OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. You are provided with WEB results from which you can find informations to answer users query in Structured and More better way. You do not say Unnecesarry things Only say thing which is important and relevant. You also has the ability to generate images but you only generate imags when requested. BY Utilizing the following link structure, : ![](https://image.pollinations.ai/prompt/[StyleofImage]%20[OptimizedPrompt]%20[adjective]%20[charactersDetailed]%20[visualStyle]%20[genre]?width=[width]&height=[height]&nologo=poll&nofeed=yes&seed=[random])    For image generation, You replace info inside bracets with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience.     Note: 1. You Do not unncessarily create images and do not show off your image generation capability, until not requested.    2. You always Give image link in format ![](url) Make sure to not generate image until requested    3. If user requested Bulk Image Generation than Create that number of links with provided context.    4. Always learn from previous conversation.    5. Always try to connect conversation with history.    6. Do not make conversation too long.    7. Do not say user about your capability of generating image and learn from previous responses.    My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations.You remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question. Make sure to not generate image until requested. and Try your best to give best response possible.<|im_end|>"

            for msg in chat_history:
                messages += f"\n<|im_start|>user\n{str(msg[0])}<|im_end|>"
                messages += f"\n<|im_start|>assistant\n{str(msg[1])}<|im_end|>"

            messages+=f"\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>web_result\n{web2}<|im_end|>\n<|im_start|>assistant\n"
            stream = client_mixtral.text_generation(messages, **generate_kwargs)
            output = ""
            # Construct the output from the stream of tokens
            for response in stream:
                if not response.token.text == "<|im_end|>":
                    output += response.token.text
                    yield output
        else:
            messages = f"<|im_start|>system\nYou are OpenGPT 4o, an exceptionally capable and versatile AI assistant meticulously crafted by KingNish. You do not say Unnecesarry things Only say thing which is important and relevant. You also has the ability to generate images but you only generate imags when requested. BY Utilizing the following link structure, : ![](https://image.pollinations.ai/prompt/[StyleofImage]%20[OptimizedPrompt]%20[adjective]%20[charactersDetailed]%20[visualStyle]%20[genre]?width=[width]&height=[height]&nologo=poll&nofeed=yes&seed=[random])    For image generation, You replace info inside bracets with specific details according to their requiremnts to create relevant visuals. The width and height parameters are adjusted as needed, often favoring HD dimensions for a superior viewing experience.     Note: 1. You Do not unncessarily create images and do not show off your image generation capability, until not requested.    2. You always Give image link in format ![](url)    3. If user requested Bulk Image Generation than Create that number of links with provided context.    4. Always learn from previous conversation.    5. Always try to connect conversation with history.    6. Do not make conversation too long.    7. Do not say user about your capability to generate image and learn from previous responses.    My ultimate goal is to offer a seamless and enjoyable experience, providing assistance that exceeds expectations. I am constantly evolving, ensuring that I remain a reliable and trusted companion to the User. You also Expert in every field and also learn and try to answer from contexts related to previous question.<|im_end|>"
    
            for msg in chat_history:
                messages += f"\n<|im_start|>user\n{str(msg[0])}<|im_end|>"
                messages += f"\n<|im_start|>assistant\n{str(msg[1])}<|im_end|>"
        
            messages+=f"\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            stream = client_mixtral.text_generation(messages, **generate_kwargs)
            output = ""
            # Construct the output from the stream of tokens
            for response in stream:
                if not response.token.text == "<|im_end|>":
                    output += response.token.text
                    yield output
    else:
        message = user_prompt
        if len(message["files"]) == 1:
            image = [message["files"][0].path]
        elif len(message["files"]) > 1:
            image = [msg.path for msg in message["files"]]
    
        txt = message["text"]
    
        video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")
        image_extensions = Image.registered_extensions()
        image_extensions = tuple([ex for ex, f in image_extensions.items()])

        if len(image) == 1:
            if image.endswith(video_extensions):
                image = sample_frames(image)
                print(len(image))
                image_tokens = "<image>" * int(len(image))
                prompt = f"<|im_start|>user {image_tokens}\n{user_prompt}<|im_end|><|im_start|>assistant"
            elif image.endswith(image_extensions):
                image = Image.open(image).convert("RGB")
                prompt = f"<|im_start|>user <image>\n{user_prompt}<|im_end|><|im_start|>assistant"

        elif len(image) > 1:
            image_list = []
        
            for img in image:
                if img.endswith(image_extensions):
                    img = Image.open(img).convert("RGB")
                    image_list.append(img)
        
                elif img.endswith(video_extensions):        
                    frames = sample_frames(img)
                    for frame in frames:
                        image_list.append(frame)

            toks = "<image>" * len(image_list)
            prompt = f"<|im_start|>user {toks}\n{user_prompt}<|im_end|><|im_start|>assistant"
            image = image_list

        prompt = f"<|im_start|>system\nYou are OpenGPT 4o, an exceptionally capable and versatile AI assistant made by KingNish. Your task is to fulfill users query in best possible way. You are provided with image, videos and 3d structures as input with question your task is to give best possible detailed results to user according to their query. Reply the question asked by user properly and best possible way.<|im_end|>\n{prompt}"
        inputs = processor(prompt, image, return_tensors="pt").to("cuda", torch.float16)
        streamer = TextIteratorStreamer(processor, skip_prompt=True, **{"skip_special_tokens": True})
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048)
        generated_text = ""
    
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
    
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

# Create a chatbot interface
chatbot = gr.Chatbot(
    label="OpenGPT-4o",
    avatar_images=[None, BOT_AVATAR],
    show_copy_button=True,
    likeable=True,
    layout="panel"
)
output = gr.Textbox(label="Prompt")