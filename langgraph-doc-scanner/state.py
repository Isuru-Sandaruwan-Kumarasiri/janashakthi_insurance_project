"""
state.py — Shared state object passed through every LangGraph node.
All nodes read from and write to this TypedDict.
"""

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any
import numpy as np


class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────
    image_path: str                         # Original file path
    raw_image: Optional[np.ndarray]         # Loaded image array (BGR)

    # ── After Preprocessor ───────────────────────────────────────────────
    preprocessed_image: Optional[np.ndarray]
    preprocessing_steps: List[str]          # e.g. ["denoise", "deskew"]

    # ── After PaddleOCR ──────────────────────────────────────────────────
    ocr_raw_text: str                       # All text joined
    ocr_blocks: List[Dict[str, Any]]        # [{text, bbox, confidence}]

    # ── After Classifier ─────────────────────────────────────────────────
    doc_type: Optional[str]                 # "id_card" | "water_bill" | "electricity_bill" | "unknown"
    classification_confidence: float

    # ── After VLM Extractor ──────────────────────────────────────────────
    extracted_fields: Dict[str, Any]        # Raw VLM extraction result
    vlm_model_used: str

    # ── After Validator ──────────────────────────────────────────────────
    final_output: Dict[str, Any]            # Final validated JSON
    validation_errors: List[str]

    # ── Meta ─────────────────────────────────────────────────────────────
    error: Optional[str]                    # Set if any node fails
    processing_time_ms: float