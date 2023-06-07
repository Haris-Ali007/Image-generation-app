from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


negative_prompt = "ugly, text, words, characters, tiling, poorly drawn hands, poorly drawn feet, \
poorly drawn face, out of frame, extra limbs, disfigured, deformed, \
body out of frame, bad anatomy, watermark, signature, cut off, low contrast, \
underexposed, overexposed, bad art, beginner, amateur, distorted face."

# def generate_image(prompt, model_id, classifier_guidance):
#     model_id = "stabilityai/stable-diffusion-2-1"
#     scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
#     pipe = StableDiffusionPipeline.from_pretrained(model_id)
#     pipe = pipe.to(torch_device)
#     image = pipe(prompt, num_inference_steps=30, negative_prompt=negative_prompt).images[0]  
#     return  image

def test_print(prompt, 
            negative_prompt,
            model_dropdown,
            scheduler_dropdown,
            inference_steps,
            guidance_scale):
    print(prompt)
    print(negative_prompt)
    print(model_dropdown)
    print(scheduler_dropdown)
    print(inference_steps)
    print(guidance_scale)
    
    