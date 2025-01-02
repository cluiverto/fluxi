import torch
from diffusers import FluxPipeline
from IPython.display import display

class Generate:
    def __init__(self, model_name="black-forest-labs/FLUX.1-dev"):
        self.model_name = model_name
        self.pipe = self.load_model(model_name)

    def load_model(self, model_name):
        # Load the base model with CPU offloading enabled
        pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()  # Enable CPU offloading
        pipe.enable_sequential_cpu_offload()  # Enable sequential CPU offloading
        
        return pipe

    def generate_image(self, prompt):
        """Generate multiple images based on a text prompt with model-specific parameters."""
        
        # Define parameters based on the selected model type
        if self.model_name == "black-forest-labs/FLUX.1-schnell":
            height = 1024
            width = 1024
            num_inference_steps = 4  # Faster generation for Schnell
            guidance_scale = 3.5
            max_sequence_length = 512
        elif self.model_name == "black-forest-labs/FLUX.1-dev":
            height = 1024  # Different height for dev
            width = 1024  # Different width for dev
            num_inference_steps = 32  # More inference steps for dev
            guidance_scale = 4.0  # Higher guidance scale for dev
            max_sequence_length = 512  # Lower max sequence length for dev

        # Generate multiple images (3 in this case)
        images = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=3  # Generate three images at once
        ).images
        
        return images