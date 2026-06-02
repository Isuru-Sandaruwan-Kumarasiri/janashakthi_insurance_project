"""
nodes/vlm_extractor.py

VLM Extractor node — uses the Vision-Language Model to extract
structured fields from the document image, guided by:
  1. The classified doc_type
  2. The raw OCR text (as additional context in prompt)

Returns extracted_fields as a dict.
"""

import json
import re
import logging
from state import AgentState
from models.vlm_client import vlm_extract, get_model_name

logger = logging.getLogger(__name__)


# ── Prompt templates per doc type ─────────────────────────────────────────────

PROMPTS = {
    "id_card": """You are a precise document extraction assistant.
Look at this ID card / identity document image carefully.
The OCR text detected is:
---
{ocr_text}
---
Extract ONLY the following fields from the document.
Respond with a valid JSON object and nothing else.
{{
  "full_name": "",
  "id_number": "",
  "date_of_birth": "",
  "gender": "",
  "nationality": "",
  "address": "",
  "issue_date": "",
  "expiry_date": "",
  "document_subtype": "national_id | passport | driving_license | other"
}}
Use null for any field not found. Do not add extra fields.""",

    "water_bill": """You are a precise document extraction assistant.
Look at this water bill image carefully.
The OCR text detected is:
---
{ocr_text}
---
Extract ONLY the following fields from the document.
Respond with a valid JSON object and nothing else.
{{
  "account_number": "",
  "customer_name": "",
  "service_address": "",
  "billing_period": "",
  "previous_reading": "",
  "current_reading": "",
  "consumption_m3": "",
  "water_charges": "",
  "sewerage_charges": "",
  "total_amount_due": "",
  "due_date": "",
  "utility_company": ""
}}
Use null for any field not found. Do not add extra fields.""",

    "electricity_bill": """You are a precise document extraction assistant.
Look at this electricity bill image carefully.
The OCR text detected is:
---
{ocr_text}
---
Extract ONLY the following fields from the document.
Respond with a valid JSON object and nothing else.
{{
  "account_number": "",
  "customer_name": "",
  "service_address": "",
  "billing_period": "",
  "previous_reading": "",
  "current_reading": "",
  "units_consumed_kwh": "",
  "energy_charges": "",
  "demand_charges": "",
  "fuel_cost_adjustment": "",
  "total_amount_due": "",
  "due_date": "",
  "utility_company": ""
}}
Use null for any field not found. Do not add extra fields.""",

    "unknown": """You are a document extraction assistant.
The document type could not be determined automatically.
The OCR text detected is:
---
{ocr_text}
---
Analyse the document and extract all key fields you can identify.
Respond with a valid JSON object with descriptive field names. Nothing else.""",
}


def run_vlm_extractor(state: AgentState) -> dict:
    """LangGraph node: extract structured fields using VLM."""
    if state.get("error"):
        return {}

    doc_type: str = state.get("doc_type", "unknown")
    ocr_text: str = state.get("ocr_raw_text", "")

    image = state.get("preprocessed_image")
    if image is None:
        image = state.get("raw_image")
    if image is None:
        return {"error": "No image for VLM extraction"}

    # Build prompt
    template = PROMPTS.get(doc_type, PROMPTS["unknown"])
    prompt = template.format(ocr_text=ocr_text[:2000])  # Truncate long OCR text

    logger.info(f"[VLM] Extracting fields for doc_type='{doc_type}' ...")

    raw_response = vlm_extract(image, prompt)
    logger.debug(f"[VLM] Raw response: {raw_response[:300]}")

    extracted = _parse_json_response(raw_response)

    return {"extracted_fields": extracted, "vlm_model_used": get_model_name()}


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _parse_json_response(response: str) -> dict:
    """
    Robustly parse JSON from VLM response.
    Handles markdown fences, trailing text, etc.
    """
    if not response:
        return {}

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Find the first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: try the whole string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("[VLM] Could not parse JSON response; returning raw text")
        return {"_raw_response": response}