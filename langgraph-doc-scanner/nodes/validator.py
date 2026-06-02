"""
nodes/validator.py

Validator node — final step in the pipeline.
  1. Checks required fields per doc type
  2. Normalises date formats and numeric strings
  3. Computes a simple field-level confidence score
  4. Assembles the final JSON output

Does NOT raise exceptions — pushes errors into validation_errors list
so downstream code can handle gracefully.
"""

import re
import logging
from datetime import datetime
from state import AgentState

logger = logging.getLogger(__name__)

# ── Required fields per doc type ─────────────────────────────────────────────
REQUIRED_FIELDS = {
    "id_card":           ["full_name", "id_number"],
    "water_bill":        ["account_number", "total_amount_due"],
    "electricity_bill":  ["account_number", "total_amount_due", "units_consumed_kwh"],
    "unknown":           [],
}


def run_validator(state: AgentState) -> dict:
    """LangGraph node: validate, normalise, and assemble final output."""
    errors: list[str] = []
    doc_type: str = state.get("doc_type", "unknown")
    fields: dict = state.get("extracted_fields", {})

    # 1. Check required fields
    required = REQUIRED_FIELDS.get(doc_type, [])
    for field in required:
        val = fields.get(field)
        if not val or val in (None, "", "null", "N/A"):
            errors.append(f"Missing required field: '{field}'")

    # 2. Normalise values
    normalised = _normalise_fields(fields)

    # 3. Compute field completeness score
    total_fields = len(normalised)
    filled_fields = sum(1 for v in normalised.values() if v not in (None, "", "null"))
    completeness = round(filled_fields / total_fields, 3) if total_fields else 0.0

    # 4. Build final output
    final_output = {
        "doc_type": doc_type,
        "confidence": state.get("classification_confidence", 0.0),
        "completeness_score": completeness,
        "fields": normalised,
        "processing_meta": {
            "ocr_engine": "paddleocr",
            "vlm_model": state.get("vlm_model_used", ""),
            "preprocessing_steps": state.get("preprocessing_steps", []),
            "ocr_blocks_count": len(state.get("ocr_blocks", [])),
        },
        "validation_errors": errors,
    }

    if errors:
        logger.warning(f"[Validator] {len(errors)} validation error(s): {errors}")
    else:
        logger.info(f"[Validator] Output valid. Completeness={completeness:.0%}")

    return {"final_output": final_output, "validation_errors": errors}


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalise_fields(fields: dict) -> dict:
    """Apply type-specific normalisation to extracted field values."""
    result = {}
    for key, value in fields.items():
        if value in (None, "null", "N/A", "n/a", ""):
            result[key] = None
            continue

        val_str = str(value).strip()

        # Dates: try to normalise to YYYY-MM-DD
        if any(k in key.lower() for k in ("date", "expiry", "issue", "period", "due")):
            result[key] = _try_normalise_date(val_str)
        # Amounts: strip currency symbols, keep numeric
        elif any(k in key.lower() for k in ("amount", "charge", "fee", "total", "payable")):
            result[key] = _clean_amount(val_str)
        else:
            result[key] = val_str

    return result


def _try_normalise_date(raw: str) -> str:
    """Best-effort date normalisation. Returns original string if parsing fails."""
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
        "%d %b %Y", "%d %B %Y", "%B %d, %Y",
        "%d/%m/%y", "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # Return as-is if no format matches


def _clean_amount(raw: str) -> str:
    """Remove currency symbols, keep digits, commas, dots."""
    cleaned = re.sub(r"[^\d.,]", "", raw).strip()
    return cleaned if cleaned else raw