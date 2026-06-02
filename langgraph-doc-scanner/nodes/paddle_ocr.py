"""
nodes/paddle_ocr.py

PaddleOCR node — runs OCR on the preprocessed image.
Returns raw text and structured block list (text + bbox + confidence).

PaddleOCR is open-source, multilingual, and runs fully locally.
Install: pip install paddlepaddle paddleocr
"""

import logging
import numpy as np
from typing import List, Dict, Any

from state import AgentState

logger = logging.getLogger(__name__)

# Lazy-load PaddleOCR once (model download on first run)
_paddle_instance = None


def _get_paddle():
    global _paddle_instance
    if _paddle_instance is None:
        from paddleocr import PaddleOCR
        _paddle_instance = PaddleOCR(
            lang="en",            # Change to "ch" for Chinese, "ms" for Malay, etc.
        )
        logger.info("[PaddleOCR] Model loaded.")
    return _paddle_instance


def run_paddle_ocr(state: AgentState) -> dict:
    """LangGraph node: run PaddleOCR on preprocessed image."""
    if state.get("error"):
        return {}  # Propagate earlier errors (already in state)

    img = state.get("preprocessed_image")
    if img is None:
        img = state.get("raw_image")
    if img is None:
        return {"error": "No image available for OCR"}

    try:
        ocr = _get_paddle()
        results = ocr.predict(img)

        blocks: List[Dict[str, Any]] = []
        text_lines: List[str] = []

        if results:
            r = results[0]  # First (and usually only) OCRResult
            rec_texts = r.get("rec_texts", []) or []
            rec_scores = r.get("rec_scores", []) or []
            dt_polys = r.get("dt_polys", []) or []

            for i, text in enumerate(rec_texts):
                conf = rec_scores[i] if i < len(rec_scores) else 0.0
                bbox = dt_polys[i] if i < len(dt_polys) else []
                # Convert numpy arrays to plain lists for JSON serialisation
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                blocks.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": round(float(conf), 4),
                })
                text_lines.append(text)

        raw_text = "\n".join(text_lines)
        logger.info(f"[PaddleOCR] Extracted {len(blocks)} blocks, {len(raw_text)} chars")
        return {"ocr_raw_text": raw_text, "ocr_blocks": blocks}

    except Exception as e:
        logger.error(f"[PaddleOCR] Error: {e}")
        return {"error": f"PaddleOCR failed: {str(e)}"}