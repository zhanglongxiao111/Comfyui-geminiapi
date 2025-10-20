import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types


class NanoBananaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": 69, "min": -1, "max": 2147483646, "step": 1},
                ),
                "image_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 5, "step": 1},
                ),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "prompt_input": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction_input": ("STRING", {"multiline": True, "default": ""}),
                "api_key_input": ("STRING", {"multiline": False, "default": ""}),
                "image_count_input": (
                    "INT",
                    {"default": 0, "min": 0, "max": 5, "step": 1},
                ),
                "use_concurrency": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "ExternalAPI/Image/Generation"

    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    def pil_to_tensor(self, images: List[Image.Image]):
        tensors = []
        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")
            array = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(array)
            tensors.append(tensor.unsqueeze(0))
        if not tensors:
            tensors.append(torch.zeros((1, 512, 512, 3), dtype=torch.float32))
        return torch.cat(tensors, dim=0)

    def _resolve_string(self, manual: str, incoming: Optional[str]) -> str:
        if isinstance(incoming, str) and incoming.strip():
            return incoming.strip()
        return (manual or "").strip()

    def _resolve_count(self, manual: int, incoming: Optional[int]) -> int:
        candidate = incoming if isinstance(incoming, int) and incoming > 0 else manual
        candidate = max(1, candidate)
        return min(candidate, 5)

    def _encode_reference_images(
        self, tensors: List[Optional[torch.Tensor]]
    ) -> List[bytes]:
        encoded: List[bytes] = []
        for tensor in tensors:
            if tensor is None:
                continue
            pil_img = self.tensor_to_pil(tensor)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            encoded.append(buffer.getvalue())
        return encoded

    def _build_contents(self, prompt: str, image_payloads: List[bytes]):
        parts: List[types.Part] = []
        for payload in image_payloads:
            parts.append(types.Part.from_bytes(mime_type="image/png", data=payload))
        prompt_text = prompt.strip()
        if prompt_text:
            parts.append(types.Part.from_text(text=prompt_text))
        if not parts:
            raise ValueError("At least one prompt or reference image must be provided.")
        return [types.Content(role="user", parts=parts)]

    def _extract_images(self, response) -> List[Image.Image]:
        images: List[Image.Image] = []
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return images
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                inline = getattr(part, "inline_data", None)
                if not inline or not getattr(inline, "data", None):
                    continue
                try:
                    image = Image.open(io.BytesIO(inline.data))
                    images.append(image.convert("RGB"))
                except Exception:
                    continue
        return images

    def _generate_once(
        self,
        api_key: str,
        aspect_ratio: str,
        temperature: float,
        top_p: float,
        seed: Optional[int],
        prompt: str,
        system_instruction: str,
        image_payloads: List[bytes],
    ) -> List[Image.Image]:
        client = genai.Client(api_key=api_key)
        contents = self._build_contents(prompt, image_payloads)

        config_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(aspect_ratio=aspect_ratio),
        }
        if isinstance(seed, int) and seed >= 0:
            config_kwargs["seed"] = seed

        config = types.GenerateContentConfig(**config_kwargs)

        instruction = system_instruction.strip()
        if instruction:
            config.system_instruction = [types.Part.from_text(text=instruction)]

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=config,
        )

        images = self._extract_images(response)
        if not images:
            print("[Nano Banana] API returned no images; using fallback frame.")
        return images

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (512, 512), color="#000000")

    def _accumulate_results(
        self,
        desired: int,
        generator,
    ) -> List[Image.Image]:
        results: List[Image.Image] = []
        errors: List[Exception] = []

        for item in generator:
            try:
                batch = item() if callable(item) else item
                results.extend(batch)
                if len(results) >= desired:
                    break
            except Exception as exc:
                errors.append(exc)
                continue

        if len(results) < desired:
            missing = desired - len(results)
            if errors:
                print(
                    f"[Nano Banana] Encountered {len(errors)} errors; "
                    "supplementing with fallback frames."
                )
            results.extend(self._blank_image() for _ in range(missing))

        if not results:
            results = [self._blank_image() for _ in range(desired)]

        return results[:desired]

    def generate(
        self,
        api_key,
        aspect_ratio,
        temperature,
        top_p,
        seed,
        image_count,
        prompt="",
        system_instruction="",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        prompt_input="",
        system_instruction_input="",
        api_key_input="",
        image_count_input=0,
        use_concurrency=False,
    ):
        resolved_key = self._resolve_string(api_key, api_key_input) or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not resolved_key:
            raise ValueError("No API key provided.")

        resolved_prompt = self._resolve_string(prompt, prompt_input)
        resolved_system = self._resolve_string(system_instruction, system_instruction_input)
        desired_count = self._resolve_count(image_count, image_count_input)

        reference_payloads = self._encode_reference_images(
            [image_1, image_2, image_3, image_4, image_5]
        )

        if not reference_payloads and not resolved_prompt:
            raise ValueError("At least one image or prompt must be provided.")

        def compute_seed(index: int) -> Optional[int]:
            if not isinstance(seed, int) or seed < 0:
                return None
            candidate = seed + index
            return min(candidate, 2147483646)

        def sequential_generator():
            for idx in range(desired_count):
                yield lambda idx=idx: self._generate_once(
                    resolved_key,
                    aspect_ratio,
                    temperature,
                    top_p,
                    compute_seed(idx),
                    resolved_prompt,
                    resolved_system,
                    reference_payloads,
                )

        def concurrent_generator():
            max_workers = min(desired_count, max((os.cpu_count() or 1), 2))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx in range(desired_count):
                    futures.append(
                        executor.submit(
                            self._generate_once,
                            resolved_key,
                            aspect_ratio,
                            temperature,
                            top_p,
                            compute_seed(idx),
                            resolved_prompt,
                            resolved_system,
                            reference_payloads,
                        )
                    )
                for future in as_completed(futures):
                    yield lambda fut=future: fut.result()

        use_parallel = bool(use_concurrency) and desired_count > 1
        generator = concurrent_generator if use_parallel else sequential_generator
        images = self._accumulate_results(desired_count, generator())

        return (self.pil_to_tensor(images),)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return (
            f"{kwargs.get('prompt', '')}-"
            f"{kwargs.get('prompt_input', '')}-"
            f"{kwargs.get('temperature', 0.5)}-"
            f"{kwargs.get('top_p', 0.85)}-"
            f"{kwargs.get('seed', 69)}-"
            f"{kwargs.get('aspect_ratio', '1:1')}-"
            f"{kwargs.get('image_count', 1)}-"
            f"{kwargs.get('image_count_input', 0)}-"
            f"{kwargs.get('use_concurrency', False)}-"
            f"{kwargs.get('image_1')}-"
            f"{kwargs.get('image_2')}-"
            f"{kwargs.get('image_3')}-"
            f"{kwargs.get('image_4')}-"
            f"{kwargs.get('image_5')}"
        )


NODE_CLASS_MAPPINGS = {"NanoBananaNode": NanoBananaNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaNode": "Nano Banana"}
