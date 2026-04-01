import numpy as np
from typing import Optional, Union
from PIL import Image
import io
import base64


class VLMCaptioner:
    """
    VLM-based image captioning for Agent A.

    Generates natural language descriptions of detected objects/scenes
    to provide semantic context beyond pure object detection.
    Uses any vision language model served via Ollama (e.g. moondream, llava).
    """

    def __init__(
        self,
        model_name: str = "moondream",
        device: str = "cpu",
        max_tokens: int = 120,
        temperature: float = 0.65,
        ollama_host: str = "http://localhost:11434",
        timeout: int = 60,
        max_image_width: int = 378,
    ):
        """
        Initialize VLM captioner
        Args:
            model_name: VLM model name as registered in Ollama
            device: Device to run inference on ('cuda' or 'cpu')
            max_tokens: Maximum tokens in generated caption
            temperature: Sampling temperature (higher = more creative)
            timeout: HTTP timeout in seconds for Ollama requests
            max_image_width: Downscale images wider than this before sending
        """
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout = timeout
        self.max_image_width = max_image_width

        print(f"Initializing VLM Captioner...")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Max tokens: {max_tokens}")

        self.model_loaded = True

        print("VLM Captioner initialized (using Ollama API)")

    def warmup(self) -> bool:
        """
        Send a lightweight request to Ollama to force model load into VRAM.
        Returns True if the model responded successfully.
        """
        import requests

        print("Warming up VLM model (loading into VRAM)...")
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {"num_predict": 1, "num_gpu": 99},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            print("VLM model warm — ready for captioning")
            return True
        except Exception as e:
            print(f"VLM warmup failed: {e}")
            return False

    def caption_detection(
        self,
        image: Union[str, np.ndarray],
        detection_bbox: tuple,
        class_name: str,
        confidence: float,
    ) -> str:
        """
        Generate contextual caption for a specific detection.

        Sends the full frame to the VLM with bbox coordinates embedded in the
        prompt so the model retains full scene context while knowing exactly
        where to focus.

        Args:
            image: Full (uncropped) frame
            detection_bbox: (x_min, y_min, x_max, y_max) in pixels
            class_name: Detected object class
            confidence: Detection confidence

        Returns:
            Pure caption string
        """
        pil_image = self._prepare_image(image)
        img_w, img_h = pil_image.size
        prompt = self._build_sar_prompt(class_name, detection_bbox, img_w, img_h)
        return self._generate_caption_ollama(pil_image, prompt)

    def _build_sar_prompt(
        self,
        class_name: str,
        bbox: Optional[tuple] = None,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> str:
        """
        Build SAR-focused prompt for a person detection.

        Uses relative position (top/middle/bottom, left/center/right) derived
        from the bbox centroid. Pixel coordinates are intentionally omitted —
        testing showed they cause moondream to return empty or garbage responses.

        Args:
            class_name: Detected object class
            bbox: Optional (x_min, y_min, x_max, y_max) in pixels
            img_w: Full image width in pixels (used for relative position)
            img_h: Full image height in pixels (used for relative position)

        Returns:
            Prompt string
        """
        if bbox is not None and img_w and img_h:
            x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            h_pos = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
            v_pos = "top" if cy < img_h / 3 else ("bottom" if cy > 2 * img_h / 3 else "middle")
            position = f"{v_pos}-{h_pos}"
        else:
            position = "center"

        return (
            f"This is an aerial drone image. "
            f"There is a {class_name} in the {position} of the frame. "
            f"Describe their clothing color and body position."
        )

    def _prepare_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')

        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]
            else:
                image_rgb = image
            return Image.fromarray(image_rgb)

        elif isinstance(image, Image.Image):
            return image.convert('RGB')

        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _generate_caption_ollama(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate caption using Ollama API.

        Args:
            image: PIL Image
            prompt: Optional custom prompt

        Returns:
            Generated caption (cleaned and trimmed)
        """
        import requests
        import json

        # Downscale large images to reduce vision-encoder work
        if image.width > self.max_image_width:
            ratio = self.max_image_width / image.width
            new_size = (self.max_image_width, int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        if prompt is None:
            prompt = (
                "In 2 sentences or less: Describe key objects, people, "
                "and environmental conditions relevant for drone search and rescue. "
                "Focus on colors, positions, and hazards only. "
                "Do not describe camera quality or image characteristics."
            )

        url = f"{self.ollama_host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_gpu": 99
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            caption = result.get("response", "").strip()

            if not caption:
                return "Error: empty response from VLM"

            return self._clean_caption(caption)

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Is it running?")
            print("   Start Ollama with: ollama serve")
            return "Error: Ollama not available"

        except requests.exceptions.Timeout:
            print("Error: Caption generation timed out")
            return "Error: Caption timeout"

        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"Error: {str(e)}"

    def _clean_caption(self, caption: str) -> str:
        """
        Clean and filter caption output.

        Removes camera quality commentary, technical image descriptions,
        and unnecessary preambles.
        """
        unwanted_phrases = [
            "the image appears to be",
            "this image shows",
            "the image is",
            "it appears to be",
            "the photo shows",
            "the picture shows",
            "captured with a",
            "taken with a",
            "smartphone",
            "low resolution",
            "high resolution",
            "the quality of",
            "image quality",
            "the camera",
            "camera flash",
            "visible in the image",
            "visible in the photo",
            "can be seen in"
        ]

        # Reject non-English output (moondream occasionally hallucinates in other languages)
        ascii_ratio = sum(1 for c in caption if ord(c) < 128) / max(len(caption), 1)
        if ascii_ratio < 0.8:
            return "Error: non-English response from VLM"

        # Reject all-punctuation garbage (e.g. "!!!!", "!!!!!")
        alnum_ratio = sum(1 for c in caption if c.isalnum()) / max(len(caption), 1)
        if alnum_ratio < 0.3:
            return "Error: garbage response from VLM"

        caption_lower = caption.lower()

        unwanted_count = sum(1 for phrase in unwanted_phrases if phrase in caption_lower)
        if unwanted_count >= 3:
            return "Scene detected, details unclear."

        import re
        for phrase in unwanted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            caption = pattern.sub("", caption)

        caption = caption.strip()
        caption = " ".join(caption.split())

        if caption:
            caption = caption[0].upper() + caption[1:]

        if caption and caption[-1] not in ".!?":
            caption += "."

        sentences = caption.split('. ')
        if len(sentences) > 2:
            caption = '. '.join(sentences[:2]) + '.'

        return caption
