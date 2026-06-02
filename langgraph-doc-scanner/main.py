"""
main.py — Entry point for the LangGraph OCR Document Agent.

Usage:
  python main.py --image path/to/doc.jpg
  python main.py --folder path/to/docs/
  python main.py --image doc.jpg --debug
"""

import argparse
import json
import logging
import time
import os
from pathlib import Path

from graph import get_graph
from state import AgentState

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Core runner ───────────────────────────────────────────────────────────────

def process_image(image_path: str, debug: bool = False) -> dict:
    """
    Run the full LangGraph OCR pipeline on a single image.
    Returns the final JSON output dict.
    """
    start = time.time()

    # Initialise state
    initial_state: AgentState = {
        "image_path": image_path,
        "raw_image": None,
        "preprocessed_image": None,
        "preprocessing_steps": [],
        "ocr_raw_text": "",
        "ocr_blocks": [],
        "doc_type": None,
        "classification_confidence": 0.0,
        "extracted_fields": {},
        "vlm_model_used": "",
        "final_output": {},
        "validation_errors": [],
        "error": None,
        "processing_time_ms": 0.0,
    }

    graph = get_graph()

    try:
        result_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return {"error": str(e), "image_path": image_path}

    elapsed_ms = round((time.time() - start) * 1000, 1)

    output = result_state.get("final_output", {})
    output["processing_time_ms"] = elapsed_ms
    output["image_path"] = image_path

    # Hard error propagation
    if result_state.get("error"):
        output["error"] = result_state["error"]

    if debug:
        output["_debug"] = {
            "ocr_raw_text":  result_state.get("ocr_raw_text", "")[:500],
            "ocr_blocks":    result_state.get("ocr_blocks", [])[:10],
            "preprocessing": result_state.get("preprocessing_steps", []),
        }

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LangGraph OCR Document Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single image file")
    group.add_argument("--folder", help="Path to folder of images")
    parser.add_argument("--debug", action="store_true", help="Include debug info in output")
    parser.add_argument("--out",   help="Write JSON output to this file")
    args = parser.parse_args()

    results = []

    if args.image:
        result = process_image(args.image, debug=args.debug)
        results.append(result)
        _print_result(result)

    elif args.folder:
        folder = Path(args.folder)
        images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        logger.info(f"Found {len(images)} image(s) in {folder}")
        for img_path in sorted(images):
            result = process_image(str(img_path), debug=args.debug)
            results.append(result)
            _print_result(result)

    # Optional: write to file
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
        logger.info(f"Output written to {args.out}")


def _print_result(result: dict):
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()