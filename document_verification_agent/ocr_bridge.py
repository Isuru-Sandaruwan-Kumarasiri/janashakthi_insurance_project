# """
# ocr_bridge.py — Bridge to the existing OCR pipeline.

# Saves uploaded image bytes to a temp file and invokes
# the OCR document_agent.process_document() function.
# """

# import logging
# import sys
# import tempfile
# import uuid
# from pathlib import Path
# from typing import Any, Dict, Optional

# logger = logging.getLogger(__name__)

# # Add the OCR directory to sys.path so we can import from it
# _OCR_DIR = Path(__file__).resolve().parent.parent / "langgraph-doc-scanner"
# if str(_OCR_DIR) not in sys.path:
#     sys.path.insert(0, str(_OCR_DIR))


# def run_ocr_on_image(image_bytes: bytes, file_suffix: str = ".jpg") -> Dict[str, Any]:
#     """
#     Run the full OCR pipeline on raw image bytes.

#     Args:
#         image_bytes: The raw bytes of the uploaded image file.
#         file_suffix: File extension (e.g., ".jpg", ".png", ".pdf").

#     Returns:
#         The structured JSON output from the OCR pipeline, containing
#         'document_type', 'extracted_data', 'validation', and 'metadata'.
#     """
#     from graph import get_graph

#     # Save bytes to a temp file
#     temp_dir = Path(tempfile.gettempdir()) / "docagent_verification"
#     temp_dir.mkdir(parents=True, exist_ok=True)

#     filename = f"{uuid.uuid4().hex}{file_suffix}"
#     filepath = temp_dir / filename

#     try:
#         filepath.write_bytes(image_bytes)
#         logger.info(f"OCR Bridge: Processing temp file {filepath} ({len(image_bytes)} bytes)")

#         # Initialise AgentState for LangGraph pipeline
#         initial_state = {
#             "image_path": str(filepath),
#             "raw_image": None,
#             "preprocessed_image": None,
#             "preprocessing_steps": [],
#             "ocr_raw_text": "",
#             "ocr_blocks": [],
#             "doc_type": None,
#             "classification_confidence": 0.0,
#             "extracted_fields": {},
#             "vlm_model_used": "",
#             "final_output": {},
#             "validation_errors": [],
#             "error": None,
#             "processing_time_ms": 0.0,
#         }

#         # Invoke the LangGraph pipeline
#         graph = get_graph()
#         result_state = graph.invoke(initial_state)

#         # Map the graph result state back to the format document_verification_agent expects
#         result = {
#             "document_type": result_state.get("doc_type") or "unknown",
#             "extracted_data": result_state.get("extracted_fields") or {},
#             "raw_ocr_text": result_state.get("ocr_raw_text", ""),
#             "errors": [],
#         }

#         # Append any hard errors or validation errors
#         if result_state.get("error"):
#             result["errors"].append(str(result_state["error"]))
#         if result_state.get("validation_errors"):
#             result["errors"].extend([str(err) for err in result_state["validation_errors"]])

#         logger.info(f"OCR Bridge: Extracted {len(result.get('extracted_data', {}))} fields, "
#                     f"raw text length: {len(result.get('raw_ocr_text', ''))}")
#         if result.get("errors"):
#             logger.error(f"OCR Bridge: Internal pipeline errors found: {result['errors']}")
#         return result

#     except Exception as e:
#         logger.error(f"OCR Bridge: Pipeline error — {e}")
#         return {
#             "document_type": "unknown",
#             "extracted_data": {},
#             "raw_ocr_text": "",
#             "errors": [str(e)],
#         }
#     finally:
#         # Clean up temp file
#         try:
#             if filepath.exists():
#                 filepath.unlink()
#         except Exception:
#             pass




"""
ocr_bridge.py — Bridge to the existing OCR pipeline.
"""

import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Add the OCR directory to sys.path so we can import from it
_OCR_DIR = Path(__file__).resolve().parent.parent / "langgraph-doc-scanner"
if str(_OCR_DIR) not in sys.path:
    sys.path.insert(0, str(_OCR_DIR))

def run_ocr_on_image(image_bytes: bytes, file_suffix: str = ".jpg") -> Dict[str, Any]:
    from graph import get_graph

    temp_dir = Path(tempfile.gettempdir()) / "docagent_verification"
    temp_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{file_suffix}"
    filepath = temp_dir / filename

    try:
        filepath.write_bytes(image_bytes)
        logger.info(f"OCR Bridge: Processing temp file {filepath} ({len(image_bytes)} bytes)")

        initial_state = {
            "image_path": str(filepath),
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
        result_state = graph.invoke(initial_state)

        result = {
            "document_type": result_state.get("doc_type") or "unknown",
            "extracted_data": result_state.get("extracted_fields") or {},
            "raw_ocr_text": result_state.get("ocr_raw_text", ""),
            "errors": [],
        }

        if result_state.get("error"):
            result["errors"].append(str(result_state["error"]))
        if result_state.get("validation_errors"):
            result["errors"].extend([str(err) for err in result_state["validation_errors"]])

        logger.info(f"OCR Bridge: Extracted {len(result.get('extracted_data', {}))} fields, "
                    f"raw text length: {len(result.get('raw_ocr_text', ''))}")
        if result.get("errors"):
            logger.error(f"OCR Bridge: Internal pipeline errors found: {result['errors']}")
        return result

    except Exception as e:
        logger.error(f"OCR Bridge: Pipeline error — {e}")
        return {
            "document_type": "unknown",
            "extracted_data": {},
            "raw_ocr_text": "",
            "errors": [str(e)],
        }
    finally:
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception:
            pass
