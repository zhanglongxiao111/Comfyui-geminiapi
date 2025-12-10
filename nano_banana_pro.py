import io
import os
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

# 检查 google-genai 版本
MIN_GENAI_VERSION = "1.0.0"
try:
    _genai_version = importlib.metadata.version("google-genai")
    _version_parts = [int(x) for x in _genai_version.split(".")[:3]]
    _min_parts = [int(x) for x in MIN_GENAI_VERSION.split(".")[:3]]
    if _version_parts < _min_parts:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"[Nano Banana Pro] google-genai 版本过低！\n"
            f"当前版本: {_genai_version}\n"
            f"最低要求: {MIN_GENAI_VERSION}\n"
            f"\n请运行以下命令升级:\n"
            f"  pip install google-genai --upgrade\n"
            f"{'='*60}\n"
        )
except importlib.metadata.PackageNotFoundError:
    raise RuntimeError(
        f"\n{'='*60}\n"
        f"[Nano Banana Pro] 未安装 google-genai！\n"
        f"\n请运行以下命令安装:\n"
        f"  pip install google-genai>={MIN_GENAI_VERSION}\n"
        f"{'='*60}\n"
    )

DEFAULT_MODEL_ID = "gemini-3-pro-image-preview"
DEFAULT_IMAGE_SIZE = "1K"


class NanoBananaProNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
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
                    {"default": 1, "min": 1, "max": 15, "step": 1},
                ),
                "model": ("STRING", {"multiline": False, "default": DEFAULT_MODEL_ID}),
                "image_size": (
                    ["1K", "2K", "4K"],
                    {"default": DEFAULT_IMAGE_SIZE},
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
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
                "prompt_input": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction_input": ("STRING", {"multiline": True, "default": ""}),
                "api_key_input": ("STRING", {"multiline": False, "default": ""}),
                "image_count_input": (
                    "INT",
                    {"default": 0, "min": 0, "max": 15, "step": 1},
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
        target_size = None
        tensors = []
        for image in images:
            if target_size is None:
                target_size = image.size
            elif image.size != target_size:
                image = image.resize(target_size, Image.LANCZOS)
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
        return min(candidate, 15)

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
        model: str,
        image_size: str,
    ) -> List[Image.Image]:
        client = genai.Client(api_key=api_key)
        contents = self._build_contents(prompt, image_payloads)

        # Build image_config only if needed; omit aspect_ratio="auto" to avoid INVALID_ARGUMENT.
        image_cfg_kwargs = {}
        if aspect_ratio and aspect_ratio != "auto":
            image_cfg_kwargs["aspect_ratio"] = aspect_ratio
        if image_size:
            image_cfg_kwargs["image_size"] = image_size
        image_config = types.ImageConfig(**image_cfg_kwargs) if image_cfg_kwargs else None

        config_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "response_modalities": ["IMAGE"],
        }
        if image_config is not None:
            config_kwargs["image_config"] = image_config
        if isinstance(seed, int) and seed >= 0:
            config_kwargs["seed"] = seed

        config = types.GenerateContentConfig(**config_kwargs)

        instruction = system_instruction.strip()
        if instruction:
            config.system_instruction = [types.Part.from_text(text=instruction)]

        print(f"[Nano Banana Pro] Calling model: {model or DEFAULT_MODEL_ID}")
        print(f"[Nano Banana Pro] ImageConfig: {image_config}")

        response = client.models.generate_content(
            model=model or DEFAULT_MODEL_ID,
            contents=contents,
            config=config,
        )

        images = self._extract_images(response)
        if not images:
            print("[Nano Banana Pro] API returned no images; using fallback frame.")
        return images

    def _blank_image(self, size: Optional[tuple[int, int]] = None) -> Image.Image:
        width, height = size or (512, 512)
        return Image.new("RGB", (width, height), color="#000000")

    def _accumulate_results(
        self,
        desired: int,
        generator,
    ) -> List[Image.Image]:
        results: List[Image.Image] = []
        errors: List[Exception] = []

        primary_size: Optional[tuple[int, int]] = None

        for item in generator:
            try:
                batch = item() if callable(item) else item
                if batch and primary_size is None:
                    primary_size = batch[0].size
                results.extend(batch)
                if len(results) >= desired:
                    break
            except Exception as exc:
                print(f"[Nano Banana Pro] Request failed: {self._describe_exception(exc)}")
                errors.append(exc)
                continue

        if len(results) < desired:
            missing = desired - len(results)
            if errors:
                print(
                    f"[Nano Banana Pro] Encountered {len(errors)} errors; "
                    "supplementing with fallback frames."
                )
            fallback_size = primary_size or (512, 512)
            results.extend(self._blank_image(fallback_size) for _ in range(missing))

        if not results:
            results = [self._blank_image() for _ in range(desired)]

        return results[:desired]

    def _describe_exception(self, exc: Exception) -> str:
        parts: List[str] = []
        try:
            parts.append(repr(exc))
        except Exception:
            parts.append(str(exc))

        status = None
        # Common locations for HTTP status on SDK errors
        for attr in ("status_code", "code"):
            status = status or getattr(exc, attr, None)
        response = getattr(exc, "response", None)
        if response is not None:
            status = status or getattr(response, "status_code", None)
        if status:
            parts.append(f"status={status}")

        message = getattr(exc, "message", None)
        if not message and getattr(exc, "args", None):
            message = exc.args[0]
        if isinstance(message, str) and message:
            parts.append(f"message={message}")

        response = getattr(exc, "response", None)
        body = None
        if response is not None:
            # Try to extract body text for API errors
            text = getattr(response, "text", None)
            content = getattr(response, "content", None)
            json_data = getattr(response, "json", None)
            if isinstance(text, str) and text:
                body = text
            elif isinstance(content, (bytes, bytearray)):
                try:
                    body = content.decode("utf-8", errors="ignore")
                except Exception:
                    body = None
            elif callable(json_data):
                try:
                    body = json_data()
                except Exception:
                    body = None
        if body:
            if isinstance(body, (dict, list)):
                import json
                try:
                    body_str = json.dumps(body)
                except Exception:
                    body_str = str(body)
            else:
                body_str = str(body).strip()
            if len(body_str) > 800:
                body_str = body_str[:800] + "...(truncated)"
            parts.append(f"body={body_str}")

        return " | ".join(parts)

    def generate(
        self,
        api_key,
        aspect_ratio,
        temperature,
        top_p,
        seed,
        image_count,
        model,
        image_size,
        prompt="",
        system_instruction="",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
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
        resolved_model = model.strip() or DEFAULT_MODEL_ID
        resolved_image_size = (image_size or DEFAULT_IMAGE_SIZE).upper()

        reference_payloads = self._encode_reference_images(
            [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                image_7,
                image_8,
                image_9,
                image_10,
                image_11,
                image_12,
                image_13,
                image_14,
            ]
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
                    resolved_model,
                    resolved_image_size,
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
                            resolved_model,
                            resolved_image_size,
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
            f"{kwargs.get('model', DEFAULT_MODEL_ID)}-"
            f"{kwargs.get('image_size', DEFAULT_IMAGE_SIZE)}-"
            f"{kwargs.get('image_1')}-"
            f"{kwargs.get('image_2')}-"
            f"{kwargs.get('image_3')}-"
            f"{kwargs.get('image_4')}-"
            f"{kwargs.get('image_5')}-"
            f"{kwargs.get('image_6')}-"
            f"{kwargs.get('image_7')}-"
            f"{kwargs.get('image_8')}-"
            f"{kwargs.get('image_9')}-"
            f"{kwargs.get('image_10')}-"
            f"{kwargs.get('image_11')}-"
            f"{kwargs.get('image_12')}-"
            f"{kwargs.get('image_13')}-"
            f"{kwargs.get('image_14')}"
        )


NODE_CLASS_MAPPINGS = {"NanoBananaProNode": NanoBananaProNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaProNode": "Nano Banana Pro"}
