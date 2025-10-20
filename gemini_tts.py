import os
import struct
import io
import torchaudio
from google.genai import Client, types

class GeminiTTSNode:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "api_key": ("STRING", {"multiline": False, "default": ""}),
            "model": (["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],),
            "voice_id": (["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Achernar", "Laomedeia", "Rasalgethi", "Algenib", "Achird", "Pulcherrima", "Gacrux", "Schedar", "Alnilam", "Sulafat", "Sadaltager", "Sadachbia", "Vindemiatrix", "Zubenelgenubi"],),                
            "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
            "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "ExternalAPI/Audio"
    
    def generate_speech(self, text, api_key, voice_id, temperature, model, seed, system_prompt=""):
        
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        
        key = api_key.strip() or os.environ.get("GEMINI_TTS_API_KEY")
        if not key:
            raise ValueError("No API key provided.")
        
        client = Client(
            api_key=key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )
        
        # Build prompt
        prompt_text = text
        if system_prompt.strip():
            prompt_text = system_prompt.strip() + ":\n\n\"" + text + "\""
        
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])]
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
                )
            ),
        )
        
        # Generate audio
        audio_data = b""
        mime_type = None
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        ):
            if (chunk.candidates and chunk.candidates[0].content and 
                chunk.candidates[0].content.parts and 
                chunk.candidates[0].content.parts[0].inline_data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                audio_data += inline_data.data
                if mime_type is None:
                    mime_type = inline_data.mime_type
        
        if not audio_data:
            raise ValueError("No audio data received from API.")
        
        if mime_type and "L16" in mime_type:
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + len(audio_data),
                b"WAVE", b"fmt ", 16, 1, 1, 24000, 48000, 2, 16,
                b"data", len(audio_data)
            )
            audio_data = header + audio_data
        
        # Decode audio from memory
        audio_buffer = io.BytesIO(audio_data)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Return in ComfyUI audio format
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice_id', '')}-{kwargs.get('temperature', 1.0)}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('system_prompt', '')}"

NODE_CLASS_MAPPINGS = {"GeminiTTSNode": GeminiTTSNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTSNode": "Gemini TTS"}