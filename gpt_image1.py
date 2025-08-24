import requests
import base64
import json
import torch
import numpy as np
from PIL import Image
import io

class GPTImageEditNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit this image"
                }),
                "api_key": ("STRING", {
                    "default": ""
                }),
                "background": (["auto", "transparent", "opaque"], {
                    "default": "auto"
                }),
                "quality": (["auto", "high", "medium", "low"], {
                    "default": "auto"
                }),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {
                    "default": "auto"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png"
                }),
                "output_compression": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "n_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_image"
    CATEGORY = "image/ai"
    
    def tensor_to_pil(self, tensor):
        if len(tensor.shape) == 3:
            array = (tensor.cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(array)
        else:
            raise ValueError(f"Expected 3D tensor, got {len(tensor.shape)}D")
    
    def pil_to_tensor(self, image):
        array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        
        return tensor
    
    def mask_to_pil(self, mask):
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        
        array = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        mask_gray = Image.fromarray(array, mode='L')
        mask_rgba = mask_gray.convert("RGBA")
        mask_rgba.putalpha(mask_gray)
        
        return mask_rgba

    def invert_mask(self, mask_image):
        """Invert mask image pixel values"""
        array = np.array(mask_image)
        inverted_array = 255 - array
        return Image.fromarray(inverted_array, mode=mask_image.mode)
    
    def pil_to_bytes(self, image, format="png"):
        """Convert PIL Image to bytes buffer"""
        buffer = io.BytesIO()
        
        if format.lower() == 'jpeg':
            image = image.convert('RGB')
        
        image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        return buffer
    
    def edit_image(self, image, prompt, api_key, background, quality, size, 
                   output_format, output_compression, n_images, mask=None):
        
        try:
            # Prepare the request
            url = "https://api.openai.com/v1/images/edits"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare form data
            files = []
            data = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "background": background,
                "n": str(n_images),
                "output_compression": str(output_compression),
                "output_format": output_format,
                "quality": quality,
                "size": size
            }
            
            # Handle batched images - send all images as reference
            batch_size = image.shape[0]
            print(f"Processing {batch_size} images as reference")
            
            for i in range(batch_size):
                # Get single image from batch
                single_image = image[i]
                pil_image = self.tensor_to_pil(single_image)
                image_buffer = self.pil_to_bytes(pil_image, "png")
                
                # Add to form data
                files.append(('image[]', (
                    f'input_{i}.png', 
                    image_buffer, 
                    'image/png'
                )))
            
            # Add mask if provided
            if mask is not None:
                pil_mask = self.mask_to_pil(mask)
                inverted_mask = self.invert_mask(pil_mask)
                mask_buffer = self.pil_to_bytes(inverted_mask, "png")
                files.append(('mask', (
                    'mask.png', 
                    mask_buffer, 
                    'image/png'
                )))
            
            # Make the API request
            response = requests.post(url, headers=headers, data=data, files=files)
            
            # Check response
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"GPT Image Edit Error: {error_msg}")
                # Return first image from batch on error
                return (image[0:1],)
            
            # Parse response
            result = response.json()
            
            # Process the first generated image
            if result['data']:
                b64_json = result['data'][0]['b64_json']
                image_bytes = base64.b64decode(b64_json)
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Convert back to tensor
                output_tensor = self.pil_to_tensor(pil_image)
                
                return (output_tensor,)
            else:
                print("No images returned from API")
                return (image[0:1],)
                
        except Exception as e:
            print(f"GPT Image Edit Error: {str(e)}")
            # Return first image from batch on error
            return (image[0:1],)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute when prompt changes
        return kwargs.get("prompt", "")

# Node registration
NODE_CLASS_MAPPINGS = {
    "GPTImageEditNode": GPTImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageEditNode": "GPT Image Edit"
}