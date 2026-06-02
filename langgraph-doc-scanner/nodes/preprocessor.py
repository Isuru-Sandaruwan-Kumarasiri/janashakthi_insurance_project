"""
nodes/preprocessor.py

Preprocessing pipeline BEFORE OCR.
Steps applied intelligently based on image quality analysis:
  1. Load & validate image
  2. Convert to grayscale
  3. Denoise (Non-Local Means)
  4. Deskew (Hough line detection)
  5. Contrast enhancement (CLAHE)
  6. Binarisation (Otsu / adaptive threshold)
  7. Resize to optimal DPI for OCR
"""

import cv2
import numpy as np
import logging
from typing import Tuple

from state import AgentState

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_DPI_WIDTH = 1800   # pixels — PaddleOCR works best around this
SKEW_THRESHOLD   = 0.5    # degrees — skip deskew if tilt is tiny


def run_preprocessor(state: AgentState) -> dict:
    """LangGraph node: load image and run full preprocessing pipeline."""
    steps_applied: list[str] = []

    # 1. Load image
    img = cv2.imread(state["image_path"])
    if img is None:
        return {"error": f"Cannot read image at: {state['image_path']}"}

    logger.info(f"[Preprocessor] Loaded image {img.shape}")

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps_applied.append("grayscale")

    # 3. Denoise
    gray = _denoise(gray)
    steps_applied.append("denoise")

    # 4. Deskew
    gray, skew_angle = _deskew(gray)
    if abs(skew_angle) > SKEW_THRESHOLD:
        steps_applied.append(f"deskew({skew_angle:.1f}°)")

    # 5. CLAHE contrast enhancement
    gray = _clahe(gray)
    steps_applied.append("clahe_contrast")

    # 6. Adaptive binarisation (helps PaddleOCR on low-contrast docs)
    binary = _binarise(gray)
    steps_applied.append("binarise")

    # 7. Resize to target width for OCR
    binary, resized = _resize_for_ocr(binary)
    if resized:
        steps_applied.append("resize_for_ocr")

    # Convert back to 3-channel so PaddleOCR & VLM accept it uniformly
    processed_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    logger.info(f"[Preprocessor] Steps applied: {steps_applied}")
    # Return only changed keys to avoid numpy truth-value issues in LangGraph merge
    return {
        "raw_image": img.copy(),
        "preprocessed_image": processed_bgr,
        "preprocessing_steps": steps_applied,
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _denoise(gray: np.ndarray) -> np.ndarray:
    """Fast Non-Local Means denoising."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct skew using Hough line transform.
    Returns corrected image and detected angle in degrees.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return gray, 0.0

    angles = []
    for line in lines[:20]:  # Use top 20 lines only
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < 45:  # Ignore steep lines
            angles.append(angle_deg)

    if not angles:
        return gray, 0.0

    median_angle = float(np.median(angles))

    if abs(median_angle) < SKEW_THRESHOLD:
        return gray, median_angle

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, scale=1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """
    Choose Otsu or adaptive threshold based on image std-dev.
    Low variance → Otsu; high variance → adaptive.
    """
    std = gray.std()
    if std < 40:
        # Low contrast — use adaptive
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
    else:
        # Good contrast — Otsu is sufficient
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _resize_for_ocr(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Scale image so width is close to TARGET_DPI_WIDTH if it's too small or too large."""
    h, w = img.shape[:2]
    if w == TARGET_DPI_WIDTH:
        return img, False
    scale = TARGET_DPI_WIDTH / w
    if 0.8 <= scale <= 1.2:   # Within 20% — skip resize
        return img, False
    new_w = TARGET_DPI_WIDTH
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, True