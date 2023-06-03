import gradio as gr
# from image_generation import generate_image
import numpy as np
from PIL import Image

def generate_image(prompt, model):
    # Your model code here
    # Use the prompt to generate an image
    # Return the generated image
    
    # For demonstration purposes, we'll just return a random image
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(image)


dropdown_options = ["model1", "model2", "model3"]
# Define the categories and their corresponding prompts
helper_prompts = {
    "Clothes": [
        "T-shirt",
        "Jeans",
        "Dress"
    ],
    "Cars": [
        "Sedan",
        "SUV",
        "Sports car"
    ],
    "Food": [
        "Pizza",
        "Burger",
        "Sushi"
    ]
}

def main():
    input = gr.Textbox(label="Enter your prompt")
    drop_down = gr.Dropdown(choices=dropdown_options, 
                            label='Model',
                            info='Select model for inference')

    category = gr.Dropdown(choices=[""] + list(helper_prompts.keys()), label="Prompt category")
    prompt = gr.Dropdown(label="Prompt")

    def update_prompt(category_selected):
        if category_selected in helper_prompts:
            prompt.choices = helper_prompts[category_selected]
            prompt.interface.visible = True
        else:
            prompt.interface.visible = False

    category.onchange = update_prompt

    output = gr.Image(label="Generated Image")
    app = gr.Interface(fn=generate_image, 
                    inputs=[prompt, drop_down],
                    outputs=output, 
                    title="Logo Generation",
                    )

    #TODO-> create a drop down for prompt selection
    app.launch()

if __name__=="__main__":
    main()