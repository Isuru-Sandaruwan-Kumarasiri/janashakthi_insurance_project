"""
nodes/classifier.py

Classify the document type from OCR raw text using keyword heuristics.
Fast, deterministic, and requires no extra model.

Supported types:
  - id_card
  - water_bill
  - electricity_bill
  - unknown
"""

import re
import logging
from state import AgentState

logger = logging.getLogger(__name__)

# ── Keyword rules ─────────────────────────────────────────────────────────────
# Each entry: (doc_type, list_of_keyword_sets, weight)
# A keyword_set is a list of strings — ALL must match for the set to fire.
# Total matched weight → classification score.

RULES = [
    # ── ID Card signals ──────────────────────────────────────────────────
    ("id_card", ["identity card", "ic number", "nric", "national id"], 4),
    ("id_card", ["passport", "date of birth", "nationality", "place of birth"], 4),
    ("id_card", ["driver", "driving licence", "driving license", "licence no"], 4),
    ("id_card", ["date of birth", "sex", "race"], 3),
    ("id_card", ["ic no", "mykad", "my kad"], 5),

    # ── Water Bill signals ────────────────────────────────────────────────
    ("water_bill", ["water bill", "water charges", "water usage"], 5),
    ("water_bill", ["cubic metre", "cubic meter", "m3", "water consumption"], 4),
    ("water_bill", ["utility", "water supply", "sewerage"], 3),
    ("water_bill", ["meter reading", "previous reading", "current reading"], 3),
    ("water_bill", ["pub", "syabas", "span", "air selangor", "pu water"], 4),

    # ── Electricity Bill signals ──────────────────────────────────────────
    ("electricity_bill", ["electricity bill", "electric bill", "power bill"], 5),
    ("electricity_bill", ["kwh", "kilowatt hour", "units consumed", "energy charges"], 5),
    ("electricity_bill", ["tenaga nasional", "tnb", "sp group", "meralco", "pln"], 5),
    ("electricity_bill", ["tariff", "demand charge", "fuel cost adjustment"], 4),
    ("electricity_bill", ["meter reading", "peak", "off-peak", "electricity supply"], 3),
]

# Shared signals (boost whichever type already leads)
SHARED_BILL_SIGNALS = [
    "account number", "account no", "billing period", "due date",
    "amount due", "total payable", "invoice", "statement",
]


def run_classifier(state: AgentState) -> dict:
    """LangGraph node: classify doc type from OCR text."""
    if state.get("error"):
        return {}

    text: str = (state.get("ocr_raw_text") or "").lower()
    if not text:
        return {"doc_type": "unknown", "classification_confidence": 0.0}

    scores: dict[str, float] = {"id_card": 0.0, "water_bill": 0.0, "electricity_bill": 0.0}

    for (doc_type, keywords, weight) in RULES:
        if any(kw in text for kw in keywords):
            scores[doc_type] += weight

    # Shared bill signal boost (small)
    if any(sig in text for sig in SHARED_BILL_SIGNALS):
        if scores["water_bill"] > 0:
            scores["water_bill"] += 1
        if scores["electricity_bill"] > 0:
            scores["electricity_bill"] += 1

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    total = sum(scores.values()) or 1

    confidence = round(best_score / total, 3)

    if best_score == 0:
        result = {"doc_type": "unknown", "classification_confidence": 0.0}
    else:
        result = {"doc_type": best_type, "classification_confidence": confidence}

    logger.info(f"[Classifier] {result['doc_type']} (conf={result['classification_confidence']}) scores={scores}")
    return result