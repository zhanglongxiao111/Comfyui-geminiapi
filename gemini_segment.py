import base64
import os
import io
import json
import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

class GeminiSegmentationNode:
    """ComfyUI Node for Gemini API Image Segmentation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segment_prompt": ("STRING", {"default": "all objects", "multiline": True}),
                "model": ("STRING", {"default": "gemini-2.5-flash", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "thinking": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "api_key": ("STRING", {"default": "", "multiline": False})
            },
            "optional": {
                "thinking_budget": ("INT", {"default": 0, "min": -1, "max": 24576, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_segmentation"
    CATEGORY = "ExternalAPI/Image/Generation"
    
    def generate_segmentation(self, image: torch.Tensor, segment_prompt: str, model: str, 
                            temperature: float, thinking: bool, seed: int, api_key: str,
                            thinking_budget: int = 0) -> tuple:
        
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Error: No API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, or provide it in the node.")
        
        img_array = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        if len(img_array.shape) == 4:
            img_array = img_array[0]  # Remove batch dimension
        if img_array.dtype in [np.float32, np.float64]:
            img_array = (img_array * 255).astype(np.uint8)
        
        original_image = Image.fromarray(img_array).convert('RGB')
        original_width, original_height = original_image.size
        
        max_size = 1024
        scale = min(max_size / original_width, max_size / original_height)
        
        if scale < 1:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            processed_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            processed_image = original_image
        
        buffer = io.BytesIO()
        processed_image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        base_prompt = f"Give the segmentation masks for {segment_prompt}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive labels."
        
        client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3, jitter=10)))
        
        parts = [
            types.Part.from_bytes(mime_type="image/png", data=image_data),
            types.Part.from_text(text=base_prompt)
        ]
        
        model_lower = model.lower()
        
        if "gemini-2.0" in model_lower:
            print("Gemini-2.0 models do not support thinking - disabling thinking config")
            final_thinking_budget = None
        elif not thinking:
            final_thinking_budget = 0
            if "gemini-2.5-pro" in model_lower:
                print("Gemini-2.5-Pro cannot have thinking turned off - defaulting thinking budget to -1")
                final_thinking_budget = -1
        else:
            final_thinking_budget = thinking_budget
            if "gemini-2.5-pro" in model_lower and final_thinking_budget == 0:
                print("Gemini-2.5-Pro cannot have thinking turned off - defaulting thinking budget to -1")
                final_thinking_budget = -1
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_mime_type="text/plain"
        )
        
        if "gemini-2.0" not in model_lower:
            config.thinking_config = types.ThinkingConfig(thinking_budget=final_thinking_budget)
        
        # Generate content
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=config
            )
            
            response_text = response.text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            
            segments = json.loads(response_text)
            
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {str(e)}")
        
        # Create mask from segments
        proc_width, proc_height = processed_image.size
        mask_image = Image.new('L', (proc_width, proc_height), 0)
        
        # Sort segments by size (largest first)
        segments_with_size = []
        for segment in segments:
            box_2d = segment['box_2d']
            ymin, xmin, ymax, xmax = box_2d
            w = (xmax - xmin) / 1000
            h = (ymax - ymin) / 1000
            segments_with_size.append((segment, w * h))
        
        segments_with_size.sort(key=lambda x: x[1], reverse=True)
        
        # Process each segment
        for i, (segment, _) in enumerate(segments_with_size):
            try:
                box_2d = segment['box_2d']
                ymin, xmin, ymax, xmax = box_2d
                
                x = int(xmin / 1000 * proc_width)
                y = int(ymin / 1000 * proc_height)
                w = int((xmax - xmin) / 1000 * proc_width)
                h = int((ymax - ymin) / 1000 * proc_height)
                
                mask_data = segment['mask']
                
                if isinstance(mask_data, str):
                    if mask_data.startswith('data:image'):
                        mask_data = mask_data.split(',')[1]
                    mask_bytes = base64.b64decode(mask_data)
                    mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
                else:
                    continue
                
                if mask_img.size != (w, h):
                    mask_img = mask_img.resize((w, h), Image.Resampling.LANCZOS)
                
                mask_array = list(mask_img.getdata())
                final_pixels = [255 if alpha > 128 else 0 for alpha in mask_array]
                segment_mask = Image.new('L', (w, h))
                segment_mask.putdata(final_pixels)
                
                if x + w <= proc_width and y + h <= proc_height and x >= 0 and y >= 0:
                    region = mask_image.crop((x, y, x + w, y + h))
                    region_pixels = list(region.getdata())
                    segment_pixels = list(segment_mask.getdata())
                    combined_pixels = [max(r, s) for r, s in zip(region_pixels, segment_pixels)]
                    combined_region = Image.new('L', (w, h))
                    combined_region.putdata(combined_pixels)
                    mask_image.paste(combined_region, (x, y))
                
            except Exception:
                continue
        
        if processed_image.size != original_image.size:
            mask_image = mask_image.resize(original_image.size, Image.Resampling.LANCZOS)
        
        # Convert PIL mask to ComfyUI mask format
        mask_array = np.array(mask_image, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add batch dimension
        
        return (mask_tensor,)

# Node mappings
NODE_CLASS_MAPPINGS = {"GeminiSegmentationNode": GeminiSegmentationNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiSegmentationNode": "Gemini Segmentation"}