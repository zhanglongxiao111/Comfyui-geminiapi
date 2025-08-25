import time
import os
import torch
import numpy as np
from PIL import Image
import tempfile
import uuid
from google import genai
from google.genai.types import GenerateVideosConfig
import cv2

class Veo3VideoGenerator:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": ([
                    "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", 
                    "us-west1", "us-west2", "us-west3", "us-west4", 
                    "northamerica-northeast1", "northamerica-northeast2", 
                    "southamerica-east1", "southamerica-west1", "africa-south1", 
                    "europe-west1", "europe-north1", "europe-west2", "europe-west3", 
                    "europe-west4", "europe-west6", "europe-west8", "europe-west9", 
                    "europe-west12", "europe-southwest1", "europe-central2", 
                    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", 
                    "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1", 
                    "asia-southeast2", "australia-southeast1", "australia-southeast2", 
                    "me-central1", "me-central2", "me-west1"
                ], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (["veo-3.0-generate-preview", "veo-3.0-fast-generate-preview","veo-3.0-generate-001", "veo-2.0-generate-001"], {"default": "veo-3.0-generate-001"}),
                "aspect_ratio": (["16:9"], {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    
    def setup_client(self, service_account_path, project_id, location):
        if service_account_path.strip():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path.strip()
        
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("Service account path is required.")
        
        if not project_id.strip():
            raise ValueError("Project ID is required.")
        
        return genai.Client(vertexai=True, project=project_id.strip(), location=location.strip())
    
    def pil_to_tensor(self, pil_image):
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(numpy_image).unsqueeze(0)

    def video_to_frames(self, video_response):
        temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4().hex}.mp4")
        
        try:
            video_bytes = None
            
            if hasattr(video_response, 'video_bytes'):
                video_bytes = video_response.video_bytes
            elif hasattr(video_response, 'data'):
                video_bytes = video_response.data
            elif hasattr(video_response.video, 'data'):
                video_bytes = video_response.video.data
            elif hasattr(video_response.video, 'video_bytes'):
                video_bytes = video_response.video.video_bytes
            elif hasattr(video_response.video, 'bytes'):
                video_bytes = video_response.video.bytes
            else:
                raise ValueError("Could not find video bytes in response")
            
            if video_bytes is None:
                raise ValueError("Video bytes are None")
            
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
            
            cap = cv2.VideoCapture(temp_video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                tensor_frame = self.pil_to_tensor(pil_frame)
                frames.append(tensor_frame)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            return torch.cat(frames, dim=0)
            
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
    def generate_video(self, project_id, location, service_account, prompt, negative_prompt, model, 
                      aspect_ratio, generate_audio, seed):
        
        try:
            client = self.setup_client(service_account, project_id, location)
            
            config_params = {
                "aspect_ratio": aspect_ratio,
                "generate_audio": generate_audio
            }
            
            if seed != -1:
                config_params["seed"] = seed
            if negative_prompt and negative_prompt.strip():
                config_params["negative_prompt"] = negative_prompt.strip()

            config = GenerateVideosConfig(**config_params)
            
            generation_params = {
                "model": model,
                "prompt": prompt,
                "config": config
            }
            
            operation = client.models.generate_videos(**generation_params)
            print(f"Operation started: {operation.name}")
            
            poll_interval = 15
            timeout_minutes = 10
            timeout_seconds = timeout_minutes * 60
            
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Video generation timed out after {timeout_minutes} minutes.")
                
                time.sleep(poll_interval)
                operation = client.operations.get(operation)
            
            if operation.response:
                generated_video = operation.response.generated_videos[0]
                frames_tensor = self.video_to_frames(generated_video)
                print(f"Video generated successfully. Extracted {frames_tensor.shape[0]} frames.")
                return (frames_tensor,)
            else:
                error_msg = f"Generation failed: {operation.error}" if operation.error else "Unknown error."
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Error in video generation process: {str(e)}"
            print(error_msg)
            empty_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_tensor,)

NODE_CLASS_MAPPINGS = {
    "Veo3VideoGenerator": Veo3VideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Veo3VideoGenerator": "Veo Text-to-Video (Vertex AI)"
}
