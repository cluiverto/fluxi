import torch
from diffusers import FluxPipeline

class Generate:
    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell"):
        self.pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_sequential_cpu_offload()

    def generate_image(self, prompt, height=768, width=1360, num_inference_steps=4):
        out = self.pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=3
        ).images
        return out