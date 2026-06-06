# =========================================================
# src/utils.py
# Common helpers
# =========================================================

from typing import Any, Dict, List, Optional
import json
import re
import pandas as pd


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in ["", "none", "nan", "null"]:
        return ""
    return re.sub(r"\s+", " ", text)


def clean_value(value: Any) -> Optional[str]:
    text = clean_text(value)
    return text if text else None


def yes_value(value: Any) -> bool:
    return clean_text(value).lower() in ["yes", "true", "1", "y", "checked"]


def parse_age(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_numeric(value: Any) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    nums = re.findall(r"\d+\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def parse_income_midpoint(value: Any) -> Optional[float]:
    text = clean_text(value).replace(",", "")
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    nums = [int(x) for x in nums]
    if len(nums) == 1:
        return float(nums[0])
    return float((nums[0] + nums[1]) / 2)


def safe_json_dumps(obj: Any, max_chars: Optional[int] = None) -> str:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n...TRUNCATED..."
    return text


def extract_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    text = str(response_text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def df_to_records(df: pd.DataFrame, max_rows: int = 20) -> List[Dict[str, Any]]:
    if df is None or len(df) == 0:
        return []
    temp = df.head(max_rows).copy()
    temp = temp.where(pd.notnull(temp), None)
    return json_safe(temp.to_dict(orient="records"))


def get_yes_fields(section_dict: Dict[str, Any]) -> List[str]:
    fields = []
    if not isinstance(section_dict, dict):
        return fields

    for key, value in section_dict.items():
        if isinstance(value, dict):
            for nested_key in get_yes_fields(value):
                fields.append(f"{key}.{nested_key}")
        elif yes_value(value):
            fields.append(str(key))

    return fields


def get_customer_summary(proposal_json: Dict[str, Any]) -> Dict[str, Any]:
    metadata = proposal_json.get("proposal_metadata", {})
    contact = proposal_json.get("customer_contact", {})
    personal = proposal_json.get("personal_details", {})
    proposal = proposal_json.get("proposal_details", {})

    return {
        "proposalno": metadata.get("proposalno"),
        "full_name": contact.get("full_name"),
        "name_with_initials": contact.get("name_with_initials"),
        "mobile": contact.get("mobile"),
        "email": contact.get("email"),
        "preferred_language": contact.get("preferred_language"),
        "age": personal.get("age"),
        "gender": personal.get("gender"),
        "marital_status": personal.get("marital_status"),
        "occupation": personal.get("occupation"),
        "monthly_income": personal.get("monthly_income"),
        "sum_insured": proposal.get("sum_insured"),
        "plan_type": proposal.get("plan_type")
    }


def build_retrieval_keywords(proposal_json: Dict[str, Any], rule_check: Dict[str, Any]) -> List[str]:
    personal = proposal_json.get("personal_details", {})
    proposal = proposal_json.get("proposal_details", {})
    habits = proposal_json.get("habits", {})
    medical = proposal_json.get("medical_history", {})
    family = proposal_json.get("family_history", {})
    additional = proposal_json.get("additional_questions", {})

    keywords = [
        personal.get("occupation"),
        personal.get("gender"),
        personal.get("monthly_income"),
        proposal.get("plan_type"),
        proposal.get("sum_insured"),
    ]

    if yes_value(habits.get("smoker")):
        keywords.extend(["smoker", "smoking"])
    if yes_value(habits.get("alcohol")):
        keywords.append("alcohol")
    if yes_value(family.get("has_family_medical_history")):
        keywords.append("family medical history")
    if yes_value(additional.get("hazardous_occupation")):
        keywords.append("hazardous occupation")
    if yes_value(additional.get("hazardous_sport")):
        keywords.append("hazardous sport")
    if yes_value(additional.get("criminal_offence")):
        keywords.append("criminal offence")
    if yes_value(additional.get("threat_on_life")):
        keywords.append("threat on life")

    keywords.extend(get_yes_fields(medical))

    for rule in rule_check.get("violated_rules", []):
        if isinstance(rule, dict):
            keywords.extend([
                rule.get("rule_id"),
                rule.get("rule_type"),
                rule.get("rule_description"),
                rule.get("condition_text"),
                rule.get("why_violated")
            ])

    return sorted(set([clean_text(k) for k in keywords if clean_text(k)]))
