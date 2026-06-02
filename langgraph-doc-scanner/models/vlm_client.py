"""
models/vlm_client.py

Abstraction layer for Vision-Language Models.

Supported backends:
  A) Qwen2-VL-7B-Instruct  — via HuggingFace transformers (local GPU/CPU)
  B) LLaVA (any variant)   — via Ollama REST API (easiest setup)
  C) OpenRouter             — via OpenRouter API (cloud, no GPU needed)

Set VLM_BACKEND in environment:
  export VLM_BACKEND=openrouter   # default — uses OPENROUTER_API_KEY
  export VLM_BACKEND=qwen2vl      # local HuggingFace
  export VLM_BACKEND=ollama
  export OLLAMA_MODEL=llava:13b   # optional, defaults to llava
"""

import os
import base64
import json
import logging
import numpy as np
import cv2

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

VLM_BACKEND = os.environ.get("VLM_BACKEND", "openrouter").lower()


# ── Public interface ──────────────────────────────────────────────────────────

def vlm_extract(image: np.ndarray, prompt: str) -> str:
    """
    Send image + prompt to the configured VLM backend.
    Returns the model's text response.
    """
    if VLM_BACKEND == "ollama":
        return _ollama_extract(image, prompt)
    elif VLM_BACKEND == "openrouter":
        return _openrouter_extract(image, prompt)
    else:
        return _qwen2vl_extract(image, prompt)


def get_model_name() -> str:
    if VLM_BACKEND == "ollama":
        return os.environ.get("OLLAMA_MODEL", "llava")
    elif VLM_BACKEND == "openrouter":
        return os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    return "Qwen2-VL-7B-Instruct"


# ── Backend C: OpenRouter (Cloud API) ─────────────────────────────────────────

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def _openrouter_extract(image: np.ndarray, prompt: str) -> str:
    """Call OpenRouter API with a vision-capable model."""
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("[OpenRouter] OPENROUTER_API_KEY not set in environment")
        return ""

    model_name = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    img_b64 = _image_to_b64(image)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        if not resp.ok:
            logger.error(f"[OpenRouter] HTTP {resp.status_code}: {resp.text[:500]}")
            return ""
        data = resp.json()

        # OpenRouter uses OpenAI-compatible response format
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()

        logger.warning(f"[OpenRouter] No choices in response: {data}")
        return ""

    except Exception as e:
        logger.error(f"[OpenRouter] Error: {e}")
        return ""


# ── Backend A: Qwen2-VL (HuggingFace) ────────────────────────────────────────

_qwen_model = None
_qwen_processor = None


def _load_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"[VLM] Loading {model_id}...")
        _qwen_processor = AutoProcessor.from_pretrained(model_id)
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",       # auto-selects GPU/CPU
        )
        logger.info("[VLM] Qwen2-VL loaded.")
    return _qwen_model, _qwen_processor


def _qwen2vl_extract(image: np.ndarray, prompt: str) -> str:
    """Run Qwen2-VL inference."""
    try:
        from qwen_vl_utils import process_vision_info
        import torch

        model, processor = _load_qwen()

        # Encode image to base64 data URL
        img_b64 = _image_to_b64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        # Trim the prompt tokens
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        logger.error(f"[Qwen2-VL] Error: {e}")
        return ""


# ── Backend B: Ollama (LLaVA) ─────────────────────────────────────────────────

def _ollama_extract(image: np.ndarray, prompt: str) -> str:
    """Call local Ollama REST API with vision model."""
    import requests

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "llava")
    img_b64 = _image_to_b64(image)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
    }

    try:
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        logger.error(f"[Ollama] Error: {e}")
        return ""


# ── Shared utility ─────────────────────────────────────────────────────────────

def _image_to_b64(image: np.ndarray) -> str:
    """Encode numpy BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode("utf-8")