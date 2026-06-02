# """
# agent.py — Document Verification Agent orchestrator.

# Coordinates the full verification workflow:
# 1. Run OCR on the uploaded ID card image.
# 2. Run OCR on the uploaded utility bill image.
# 3. Compare OCR results against the proposal form data using pure Python logic.
# 4. Return a structured verification report.
# """

# import logging
# from typing import Any, Dict

# from .ocr_bridge import run_ocr_on_image
# from .llm_comparator import compare_documents

# logger = logging.getLogger(__name__)


# def verify_documents(
#     proposal_data: Dict[str, Any],
#     id_card_bytes: bytes,
#     id_card_suffix: str,
#     utility_bill_bytes: bytes,
#     utility_bill_suffix: str,
# ) -> Dict[str, Any]:
#     """
#     Full document verification pipeline.

#     Args:
#         proposal_data: The complete proposal form JSON from the frontend.
#         id_card_bytes: Raw bytes of the uploaded ID card image.
#         id_card_suffix: File extension of the ID card (e.g., ".jpg").
#         utility_bill_bytes: Raw bytes of the uploaded utility bill image.
#         utility_bill_suffix: File extension of the utility bill (e.g., ".png").

#     Returns:
#         Structured verification report:
#         {
#             "overall_status": "PASS" | "FAIL" | "REVIEW",
#             "summary": "...",
#             "id_card_ocr": { ... },       # raw OCR output for reference
#             "utility_bill_ocr": { ... },  # raw OCR output for reference
#             "checks": [
#                 {
#                     "field": "Full Name",
#                     "proposal_value": "...",
#                     "document_value": "...",
#                     "source": "ID Card",
#                     "match": true/false,
#                     "confidence": 0.95,
#                     "reasoning": "..."
#                 },
#                 ...
#             ]
#         }
#     """
#     logger.info("=" * 60)
#     logger.info("Document Verification Agent — Starting")
#     logger.info("=" * 60)

#     # Step 1: OCR the ID card
#     logger.info("[Step 1/3] Running OCR on ID card...")
#     id_card_ocr = run_ocr_on_image(id_card_bytes, id_card_suffix)
#     logger.info(f"  ID Card — detected type: {id_card_ocr.get('document_type', 'unknown')}")
#     logger.info(f"  ID Card — extracted_data keys: {list(id_card_ocr.get('extracted_data', {}).keys())}")
#     logger.info(f"  ID Card — raw_ocr_text length: {len(id_card_ocr.get('raw_ocr_text', ''))}")
#     logger.info(f"  ID Card — errors: {id_card_ocr.get('errors', [])}")

#     # Step 2: OCR the utility bill
#     logger.info("[Step 2/3] Running OCR on utility bill...")
#     utility_bill_ocr = run_ocr_on_image(utility_bill_bytes, utility_bill_suffix)
#     logger.info(f"  Utility Bill — detected type: {utility_bill_ocr.get('document_type', 'unknown')}")
#     logger.info(f"  Utility Bill — extracted_data keys: {list(utility_bill_ocr.get('extracted_data', {}).keys())}")
#     logger.info(f"  Utility Bill — raw_ocr_text length: {len(utility_bill_ocr.get('raw_ocr_text', ''))}")
#     logger.info(f"  Utility Bill — errors: {utility_bill_ocr.get('errors', [])}")

#     # Step 3: LLM comparison
#     logger.info("[Step 3/3] Running LLM-based comparison...")
#     verification_result = compare_documents(
#         proposal_data=proposal_data,
#         id_card_ocr=id_card_ocr,
#         utility_bill_ocr=utility_bill_ocr,
#     )

#     # Attach FULL OCR outputs for transparency (so frontend can show them for debugging)
#     verification_result["id_card_ocr"] = {
#         "document_type": id_card_ocr.get("document_type", "unknown"),
#         "extracted_data": id_card_ocr.get("extracted_data", {}),
#         "raw_ocr_text": id_card_ocr.get("raw_ocr_text", ""),
#         "errors": id_card_ocr.get("errors", []),
#     }
#     verification_result["utility_bill_ocr"] = {
#         "document_type": utility_bill_ocr.get("document_type", "unknown"),
#         "extracted_data": utility_bill_ocr.get("extracted_data", {}),
#         "raw_ocr_text": utility_bill_ocr.get("raw_ocr_text", ""),
#         "errors": utility_bill_ocr.get("errors", []),
#     }

#     logger.info(f"Verification complete — overall status: {verification_result.get('overall_status')}")
#     logger.info("=" * 60)

#     return verification_result





"""
agent.py — Document Verification Agent orchestrator.
"""

import logging
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

from .ocr_bridge import run_ocr_on_image
from .llm_comparator import compare_documents

logger = logging.getLogger(__name__)

def verify_documents(
    proposal_data: Dict[str, Any],
    id_card_bytes: bytes,
    id_card_suffix: str,
    utility_bill_bytes: bytes,
    utility_bill_suffix: str,
) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Document Verification Agent — Starting Concurrent OCR")
    logger.info("=" * 60)

    # Step 1 & 2: Run OCR on BOTH images concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        logger.info("Submitting ID Card and Utility Bill to LangGraph simultaneously...")
        
        id_future = executor.submit(run_ocr_on_image, id_card_bytes, id_card_suffix)
        bill_future = executor.submit(run_ocr_on_image, utility_bill_bytes, utility_bill_suffix)
        
        id_card_ocr = id_future.result()
        utility_bill_ocr = bill_future.result()

    logger.info(f"ID Card OCR Complete. Found fields: {list(id_card_ocr.get('extracted_data', {}).keys())}")
    logger.info(f"Utility Bill OCR Complete. Found fields: {list(utility_bill_ocr.get('extracted_data', {}).keys())}")

    # Step 3: LLM comparison
    logger.info("[Step 3/3] Running LLM-based comparison...")
    verification_result = compare_documents(
        proposal_data=proposal_data,
        id_card_ocr=id_card_ocr,
        utility_bill_ocr=utility_bill_ocr,
    )

    # Attach FULL OCR outputs for frontend debugging
    verification_result["id_card_ocr"] = {
        "document_type": id_card_ocr.get("document_type", "unknown"),
        "extracted_data": id_card_ocr.get("extracted_data", {}),
        "raw_ocr_text": id_card_ocr.get("raw_ocr_text", ""),
        "errors": id_card_ocr.get("errors", []),
    }
    verification_result["utility_bill_ocr"] = {
        "document_type": utility_bill_ocr.get("document_type", "unknown"),
        "extracted_data": utility_bill_ocr.get("extracted_data", {}),
        "raw_ocr_text": utility_bill_ocr.get("raw_ocr_text", ""),
        "errors": utility_bill_ocr.get("errors", []),
    }

    logger.info(f"Verification complete — overall status: {verification_result.get('overall_status')}")
    logger.info("=" * 60)

    return verification_result