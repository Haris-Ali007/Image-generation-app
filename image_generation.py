from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler
import torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

schedulers = {
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler, 
    "LMSDiscreteScheduler": LMSDiscreteScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
}

negative_prompt = "ugly, text, words, characters, tiling, poorly drawn hands, poorly drawn feet, \
poorly drawn face, out of frame, extra limbs, disfigured, deformed, \
body out of frame, bad anatomy, watermark, signature, cut off, low contrast, \
underexposed, overexposed, bad art, beginner, amateur, distorted face."

def generate_image(prompt, 
                    negative_prompt,
                    model_id,
                    scheduler_name,
                    inference_steps,
                    guidance_scale):

    scheduler = schedulers[scheduler_name].from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = scheduler
    pipe = pipe.to(torch_device)
    prompts = [prompt] * 4
    negative_prompts = [negative_prompt] * 4
    images = pipe(prompts, num_inference_steps=inference_steps, negative_prompt=negative_prompts, guidance_scale=float(guidance_scale)).images  
    return  images


# def test_gui(prompt, 
#             negative_prompt,
#             model_dropdown,
#             scheduler_dropdown,
#             inference_steps,
#             guidance_scale):
#     print(prompt)
#     print(negative_prompt)
#     print(model_dropdown)
#     print(scheduler_dropdown)
#     print(inference_steps)
#     print(guidance_scale)

#     from PIL import Image
#     import numpy as np

#     # Create four noise images
#     noise_images = []
#     for _ in range(4):
#         # Generate random noise array
#         width, height = 512, 512  # Adjust as desired
#         noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
#         noise_image = Image.fromarray(noise)
#         noise_images.append(noise_image)

#     return noise_images
    