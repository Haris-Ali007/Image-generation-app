import gradio as gr
from image_generation import generate_image
# from image_generation import test_print

model_options = ["CompVis/stable-diffusion-v1-4", 
                    "runwayml/stable-diffusion-v1-5",
                    "stabilityai/stable-diffusion-2-1", 
                    "stabilityai/stable-diffusion-2-1-base",
                    "prompthero/openjourney-v4"]

scheduler_options = ["None", 
                     "EulerDiscreteScheduler", 
                     "PNDMScheduler", 
                     "DPMSolverMultistepScheduler", 
                     "LMSDiscreteScheduler", 
                     "HeunDiscreteScheduler"]

def main():
        prompt = gr.Textbox(label="Enter your prompt")
        negative_prompt = gr.Textbox(label="Negative prompt", type='text', value="disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w") 
        guidance_scale = gr.Textbox(label="Guidance scale", type='text', value=7.5)
        model_dropdown = gr.Dropdown(choices=model_options, 
                                label='Model',
                                value='prompthero/openjourney-v4',
                                info='Select model for inference')

        scheduler_dropdown = gr.Dropdown(choices=scheduler_options, 
                                label='Scheduler',
                                value='None',
                                info='Select scheduler for inference. None for default')

        inference_steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps", info="Choose between 10 and 50")
        num_of_images = gr.Slider(1, 4, value=1, step=1, label="Number of images", info="Choose between 1 and 4")

        gen_image_grid = gr.Gallery(label='Generated images').style(columns=[2], rows=[2], object_fit="cover", height="auto", preview=True)
        app = gr.Interface(fn=generate_image, 
                        inputs=[prompt, 
                                negative_prompt,
                                model_dropdown,
                                scheduler_dropdown,
                                inference_steps,
                                guidance_scale,
                                num_of_images],
                        # outputs=generated_image,
                        outputs=gen_image_grid, 
                        title="Image Generation",
                        )
        app.launch(True)

if __name__=="__main__":
        main()
    
