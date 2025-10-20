import replicate
import os
import requests
import torch
import numpy as np
from PIL import Image
import io

class FluxKontextProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Make this a 90s cartoon"
                }),
                "api_key": ("STRING", {
                    "default": ""
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "2:1", "1:2", "match_input_image"], {
                    "default": "match_input_image"
                }),
                "output_format": (["jpg", "png"], {
                    "default": "jpg"
                }),
                "safety_tolerance": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 6,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "ExternalAPI/Image/Edit"
    
    def generate_image(self, image, prompt, api_key, aspect_ratio, output_format, safety_tolerance):
        try:
            os.environ["REPLICATE_API_TOKEN"] = api_key
            
            # Convert tensor to PIL and save to buffer
            tensor = image.squeeze(0) if len(image.shape) == 4 else image
            if tensor.max() <= 1.0:
                tensor = (tensor * 255).clamp(0, 255).byte()
            pil_image = Image.fromarray(tensor.cpu().numpy(), 'RGB')
            
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Run Replicate model
            output = replicate.run(
                "black-forest-labs/flux-kontext-pro",
                input={
                    "prompt": prompt,
                    "input_image": img_buffer,
                    "aspect_ratio": aspect_ratio,
                    "output_format": output_format,
                    "safety_tolerance": safety_tolerance
                }
            )
            
            # Get URL from output
            output_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) and output else str(output))
            
            # Download and convert back to tensor
            response = requests.get(output_url, timeout=30)
            response.raise_for_status()
            
            downloaded_image = Image.open(io.BytesIO(response.content))
            if downloaded_image.mode != 'RGB':
                downloaded_image = downloaded_image.convert('RGB')
            
            np_image = np.array(downloaded_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(np_image).unsqueeze(0)
            
            return (output_tensor,)
            
        except Exception as e:
            return (torch.zeros((1, 512, 512, 3)),)

NODE_CLASS_MAPPINGS = {
    "FluxKontextProNode": FluxKontextProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextProNode": "Flux Kontext Pro"
}