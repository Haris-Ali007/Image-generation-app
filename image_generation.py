from PIL import Image
import numpy as np

def generate_image(prompt, model):
    # Your model code here
    # Use the prompt to generate an image
    # Return the generated image
    
    # For demonstration purposes, we'll just return a random image
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(image)