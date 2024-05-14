import gradio as gr

# Define the classification function
def classify_task(prompt):
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
