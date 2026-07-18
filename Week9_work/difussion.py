import torch
from diffusers import StableDiffusionPipeline

# 1. Load the pipeline in half-precision (float16)
model_id = "runwayml/stable-diffusion-v1-5"
print("Loading Stable Diffusion Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 2. Define your creative prompt
prompt = "A high-tech research lab with glowing holographic matrices, cinematic lighting, photorealistic, 8k resolution"

# 3. Run the inference pipeline
print("Generating image...")
with torch.inference_mode():
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# 4. Save the output
image.save("generated_lab.png")
print("Image successfully saved as 'generated_lab.png'!")