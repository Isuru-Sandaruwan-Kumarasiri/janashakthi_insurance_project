# """
# llm_comparator.py — LLM-based document verification.

# Uses LangChain + OpenRouter to compare OCR-extracted data from
# uploaded documents against the proposal form fields.
# """

# import json
# import logging
# import os
# import re
# from typing import Any, Dict

# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage

# load_dotenv()

# logger = logging.getLogger(__name__)

# # ─────────────────────────────────────────────
# # System prompt for document verification
# # ─────────────────────────────────────────────

# VERIFICATION_SYSTEM_PROMPT = """You are a document verification specialist for an insurance company.
# You will be given:
# 1. Proposal form data (filled in by the applicant)
# 2. OCR-extracted data from an uploaded ID card (structured data + raw OCR text)
# 3. OCR-extracted data from an uploaded utility bill (structured data + raw OCR text)

# Your task is to compare the information across these sources and verify consistency.

# You MUST perform the following checks:

# **ID Card vs Proposal Form:**
# - Compare the full name on the ID card with the name on the proposal form.
# - Compare the NIC/ID number on the ID card with the NIC on the proposal form.
# - Compare the date of birth on the ID card with the DOB on the proposal form.

# **Utility Bill vs Proposal Form:**
# - Compare the address on the utility bill with the correspondence address on the proposal form.
# - Compare the customer name on the utility bill with the name on the proposal form.

# **IMPORTANT — Matching Rules:**

# LOOSE matching (be lenient):
# - **NIC / ID Number**: Match the digits and letters only. IGNORE spaces, dashes, dots, and case. Example: "200012345678" matches "2000-1234-5678".
# - **Full Name**: IGNORE uppercase/lowercase. IGNORE word ordering. Ignore initials vs full name. Example: "K. ISURU SANDARUWAN" matches "Isuru Sandaruwan Kumarasiri". As long as the core name parts overlap, it is a MATCH.

# STRICT matching (be precise):
# - **Address**: The address on the utility bill must semantically match the proposal address. Allow abbreviations ("Rd" vs "Road", "St" vs "Street", "No." vs "Number") but the actual location, street name, city, and postal code should be the same. Different locations = mismatch.
# - **Date of Birth**: Compare actual date values. Allow format differences (DD/MM/YYYY vs YYYY-MM-DD) but the date itself must match exactly.

# **Fallback to Raw Text**: If the "Structured Data" section is empty or missing a field, you MUST search the "Raw OCR Text" to find the information. Only mark a field as "NOT_FOUND" if it cannot be found in EITHER the structured data OR the raw text.

# **Scoring (out of 100):**
# - NIC/ID Number match: 30 points
# - Full Name match: 25 points
# - Address match: 25 points
# - Date of Birth match: 20 points

# Return ONLY a valid JSON object in this exact format:
# {
#     "overall_status": "PASS" or "FAIL",
#     "overall_score": 85,
#     "summary": "Brief overall summary explaining what matched and what did not",
#     "checks": [
#         {
#             "field": "NIC Number",
#             "proposal_value": "value from proposal",
#             "document_value": "value found in document",
#             "source": "ID Card" or "Utility Bill",
#             "match": true or false,
#             "score": 30,
#             "max_score": 30,
#             "reasoning": "Detailed explanation of why this is a match or mismatch"
#         }
#     ]
# }

# Rules for overall_status:
# - "PASS": overall_score is 80 or above.
# - "FAIL": overall_score is below 80.

# When overall_status is "FAIL", the summary MUST clearly explain which specific fields failed and why, so the user knows exactly what to fix.

# Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


# def _build_comparison_prompt(
#     proposal_data: Dict[str, Any],
#     id_card_ocr: Dict[str, Any],
#     utility_bill_ocr: Dict[str, Any],
# ) -> str:
#     """Build the human message prompt with all three data sources."""

#     # Extract the relevant proposal fields
#     main_life = proposal_data.get("main_life", {})
#     proposal_summary = {
#         "full_name": main_life.get("full_name", ""),
#         "name_with_initials": main_life.get("name_with_initials", ""),
#         "nic": main_life.get("nic", ""),
#         "date_of_birth": main_life.get("dob", ""),
#         "correspondence_address": main_life.get("correspondence_address", ""),
#         "email": main_life.get("email", ""),
#         "mobile_phone": main_life.get("mobile_phone", ""),
#     }

#     # Extract relevant OCR fields
#     id_extracted = id_card_ocr.get("extracted_data", {})
#     id_raw_text = id_card_ocr.get("raw_ocr_text", "")
    
#     bill_extracted = utility_bill_ocr.get("extracted_data", {})
#     bill_raw_text = utility_bill_ocr.get("raw_ocr_text", "")

#     prompt = f"""Please verify the following documents against the proposal form data.

# === PROPOSAL FORM DATA ===
# {json.dumps(proposal_summary, indent=2)}

# === ID CARD (OCR Extracted) ===
# Document Type Detected: {id_card_ocr.get('document_type', 'unknown')}
# Structured Data:
# {json.dumps(id_extracted, indent=2)}
# Raw OCR Text (Fallback):
# {id_raw_text}

# === UTILITY BILL (OCR Extracted) ===
# Document Type Detected: {utility_bill_ocr.get('document_type', 'unknown')}
# Structured Data:
# {json.dumps(bill_extracted, indent=2)}
# Raw OCR Text (Fallback):
# {bill_raw_text}

# Now compare these three sources and return the verification JSON."""

#     return prompt


# def compare_documents(
#     proposal_data: Dict[str, Any],
#     id_card_ocr: Dict[str, Any],
#     utility_bill_ocr: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Use an LLM to compare OCR-extracted document data against the proposal form.
#     """
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         return {
#             "overall_status": "FAIL",
#             "summary": "OPENROUTER_API_KEY is not configured. Cannot run LLM verification.",
#             "checks": [],
#         }

#     models_to_try = [
#         "google/gemini-2.5-flash",
#         "google/gemini-3.5-flash",
#         "meta-llama/llama-3.3-70b-instruct:free",
#         "meta-llama/llama-3.2-3b-instruct:free",
#     ]

#     human_prompt = _build_comparison_prompt(proposal_data, id_card_ocr, utility_bill_ocr)
#     messages = [
#         SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
#         HumanMessage(content=human_prompt),
#     ]

#     last_error = None
#     for model_name in models_to_try:
#         try:
#             logger.info(f"LLM Comparator: Sending verification request to {model_name}")
#             llm = ChatOpenAI(
#                 model=model_name,
#                 openai_api_key=api_key,
#                 openai_api_base="https://openrouter.ai/api/v1",
#                 temperature=0.0,
#                 max_tokens=2000,
#                 default_headers={
#                     "HTTP-Referer": "https://github.com/langchain-ai/langchain",
#                     "X-Title": "Janashakthi Document Verification Agent",
#                 }
#             )

#             response = llm.invoke(messages)
#             raw_text = response.content.strip()

#             # Parse JSON from the response (handle markdown code blocks)
#             json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#                 logger.info(f"LLM Comparator: Verification complete using {model_name} — status: {result.get('overall_status')}")
#                 return result
#             else:
#                 logger.error(f"LLM Comparator ({model_name}): Could not parse JSON: {raw_text[:200]}")
#                 return {
#                     "overall_status": "REVIEW",
#                     "summary": f"LLM response could not be parsed. Manual review required. Model: {model_name}",
#                     "checks": [],
#                     "raw_response": raw_text,
#                 }
#         except Exception as e:
#             error_msg = str(e)
#             logger.warning(f"LLM Comparator ({model_name}): Failed — {error_msg}")
#             last_error = error_msg
#             if "429" in error_msg or "404" in error_msg or "400" in error_msg or "rate-limited" in error_msg.lower():
#                 continue # Try the next model in the list
#             else:
#                 continue # Try next model anyway for robustness

#     logger.error(f"LLM Comparator: All fallback models failed. Last error: {last_error}")
#     return {
#         "overall_status": "FAIL",
#         "summary": f"LLM verification failed after trying multiple models: {last_error}",
#         "checks": [],
#     }






"""
llm_comparator.py — LLM-based document verification.
"""

import json
import logging
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

logger = logging.getLogger(__name__)

VERIFICATION_SYSTEM_PROMPT = """You are a document verification specialist for an insurance company.
You will be given:
1. Proposal form data (filled in by the applicant)
2. OCR-extracted data from an uploaded ID card (structured data + raw OCR text)
3. OCR-extracted data from an uploaded utility bill (structured data + raw OCR text)

Your task is to compare the information across these sources and verify consistency.

You MUST perform the following checks:

**ID Card vs Proposal Form:**
- Compare the full name on the ID card with the name on the proposal form.
- Compare the NIC/ID number on the ID card with the NIC on the proposal form.
- Compare the date of birth on the ID card with the DOB on the proposal form.

**Utility Bill vs Proposal Form:**
- Compare the address on the utility bill with the correspondence address on the proposal form.
- Compare the customer name on the utility bill with the name on the proposal form.

**IMPORTANT — Matching Rules:**

LOOSE matching (be lenient):
- **NIC / ID Number**: Match the digits and letters only. IGNORE spaces, dashes, dots, and case. Example: "200012345678" matches "2000-1234-5678".
- **Full Name**: IGNORE uppercase/lowercase. IGNORE word ordering. Ignore initials vs full name. Example: "K. ISURU SANDARUWAN" matches "Isuru Sandaruwan Kumarasiri". As long as the core name parts overlap, it is a MATCH.

STRICT matching (be precise):
- **Address**: The address on the utility bill must semantically match the proposal address. Allow abbreviations ("Rd" vs "Road", "St" vs "Street", "No." vs "Number") but the actual location, street name, city, and postal code should be the same. Different locations = mismatch.
- **Date of Birth**: Compare actual date values. Allow format differences (DD/MM/YYYY vs YYYY-MM-DD) but the date itself must match exactly.

**Fallback to Raw Text**: If the "Structured Data" section is empty or missing a field, you MUST search the "Raw OCR Text" to find the information. Only mark a field as "NOT_FOUND" if it cannot be found in EITHER the structured data OR the raw text.

**Scoring (out of 100):**
- NIC/ID Number match: 30 points
- Full Name match: 25 points
- Address match: 25 points
- Date of Birth match: 20 points

Return ONLY a valid JSON object in this exact format:
{
    "overall_status": "PASS" or "FAIL",
    "overall_score": 85,
    "summary": "Brief overall summary explaining what matched and what did not",
    "checks": [
        {
            "field": "NIC Number",
            "proposal_value": "value from proposal",
            "document_value": "value found in document",
            "source": "ID Card" or "Utility Bill",
            "match": true or false,
            "score": 30,
            "max_score": 30,
            "reasoning": "Detailed explanation of why this is a match or mismatch"
        }
    ]
}

Rules for overall_status:
- "PASS": overall_score is 80 or above.
- "FAIL": overall_score is below 80.

When overall_status is "FAIL", the summary MUST clearly explain which specific fields failed and why, so the user knows exactly what to fix.

Return ONLY the JSON object. No markdown, no explanation outside the JSON."""

def _build_comparison_prompt(
    proposal_data: Dict[str, Any],
    id_card_ocr: Dict[str, Any],
    utility_bill_ocr: Dict[str, Any],
) -> str:

    main_life = proposal_data.get("main_life", {})
    proposal_summary = {
        "full_name": main_life.get("full_name", ""),
        "name_with_initials": main_life.get("name_with_initials", ""),
        "nic": main_life.get("nic", ""),
        "date_of_birth": main_life.get("dob", ""),
        "correspondence_address": main_life.get("correspondence_address", ""),
        "email": main_life.get("email", ""),
        "mobile_phone": main_life.get("mobile_phone", ""),
    }

    id_extracted = id_card_ocr.get("extracted_data", {})
    id_raw_text = id_card_ocr.get("raw_ocr_text", "")
    
    bill_extracted = utility_bill_ocr.get("extracted_data", {})
    bill_raw_text = utility_bill_ocr.get("raw_ocr_text", "")

    prompt = f"""Please verify the following documents against the proposal form data.

=== PROPOSAL FORM DATA ===
{json.dumps(proposal_summary, indent=2)}

=== ID CARD (OCR Extracted) ===
Document Type Detected: {id_card_ocr.get('document_type', 'unknown')}
Structured Data:
{json.dumps(id_extracted, indent=2)}
Raw OCR Text (Fallback):
{id_raw_text}

=== UTILITY BILL (OCR Extracted) ===
Document Type Detected: {utility_bill_ocr.get('document_type', 'unknown')}
Structured Data:
{json.dumps(bill_extracted, indent=2)}
Raw OCR Text (Fallback):
{bill_raw_text}

Now compare these three sources and return the verification JSON."""

    return prompt

def compare_documents(
    proposal_data: Dict[str, Any],
    id_card_ocr: Dict[str, Any],
    utility_bill_ocr: Dict[str, Any],
) -> Dict[str, Any]:

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "overall_status": "FAIL",
            "summary": "OPENROUTER_API_KEY is not configured. Cannot run LLM verification.",
            "checks": [],
        }

    models_to_try = [
        "google/gemini-2.5-flash",
        "google/gemini-3.5-flash",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]

    human_prompt = _build_comparison_prompt(proposal_data, id_card_ocr, utility_bill_ocr)
    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"LLM Comparator: Sending verification request to {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=2000,
                default_headers={
                    "HTTP-Referer": "https://github.com/langchain-ai/langchain",
                    "X-Title": "Janashakthi Document Verification Agent",
                }
            )

            response = llm.invoke(messages)
            raw_text = response.content.strip()

            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"LLM Comparator: Verification complete using {model_name} — status: {result.get('overall_status')}")
                return result
            else:
                logger.error(f"LLM Comparator ({model_name}): Could not parse JSON: {raw_text[:200]}")
                return {
                    "overall_status": "REVIEW",
                    "summary": f"LLM response could not be parsed. Manual review required. Model: {model_name}",
                    "checks": [],
                    "raw_response": raw_text,
                }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"LLM Comparator ({model_name}): Failed — {error_msg}")
            last_error = error_msg
            if "429" in error_msg or "404" in error_msg or "400" in error_msg or "rate-limited" in error_msg.lower():
                continue 
            else:
                continue 

    logger.error(f"LLM Comparator: All fallback models failed. Last error: {last_error}")
    return {
        "overall_status": "FAIL",
        "summary": f"LLM verification failed after trying multiple models: {last_error}",
        "checks": [],
    }