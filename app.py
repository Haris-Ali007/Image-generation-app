import gradio as gr
from image_generation import generate_image

dropdown_options = ["CompVis/stable-diffusion-v1-4", 
                    "runwayml/stable-diffusion-v1-5",
                    "stabilityai/stable-diffusion-2-1", 
                    "stabilityai/stable-diffusion-2-1-base",
                    "prompthero/openjourney-v4"]

# Define the categories and their corresponding prompts
# helper_prompts = {
#     "Clothes": [
#         "T-shirt",
#         "Jeans",
#         "Dress"
#     ],
#     "Cars": [
#         "Sedan",
#         "SUV",
#         "Sports car"
#     ],
#     "Food": [
#         "Pizza",
#         "Burger",
#         "Sushi"
#     ]
# }

# category = gr.Dropdown(choices=[""] + list(helper_prompts.keys()), label="Prompt category")
# prompt = gr.Dropdown(label="Prompt")

# def update_prompt(category_selected):
#     if category_selected in helper_prompts:
#         prompt.choices = helper_prompts[category_selected]
#         prompt.interface.visible = True
#     else:
#         prompt.interface.visible = False

# category.onchange = update_prompt

def main():
    input = gr.Textbox(label="Enter your prompt")
    drop_down = gr.Dropdown(choices=dropdown_options, 
                            label='Model',
                            info='Select model for inference')

    output = gr.Image(label="Generated Image")
    app = gr.Interface(fn=generate_image, 
                    inputs=[input, drop_down],
                    outputs=output, 
                    title="Logo Generation",
                    )

    app.launch()

if __name__=="__main__":
    main()