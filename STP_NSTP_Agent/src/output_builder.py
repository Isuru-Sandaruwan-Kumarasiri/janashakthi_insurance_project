# =========================================================
# src/output_builder.py
# Final output builder for LLM-first STP/NSTP Rule RAG Agent
# =========================================================

from typing import Any, Dict, List, Optional

from src.config import VALID_DECISIONS
from src.utils import clean_text, get_customer_summary


def get_proposal_no(proposal_json: Dict[str, Any]) -> Optional[str]:
    return proposal_json.get("proposal_metadata", {}).get("proposalno")


def dedupe_text_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    cleaned = [clean_text(x) for x in items if clean_text(x)]
    return sorted(set(cleaned))


def dedupe_rules(rules: Any) -> List[Dict[str, Any]]:
    if not isinstance(rules, list):
        return []

    final = []
    seen = set()

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        key = (
            clean_text(rule.get("rule_id")),
            clean_text(rule.get("rule_description")),
            clean_text(rule.get("why_violated"))
        )

        if key in seen:
            continue

        seen.add(key)
        final.append(rule)

    return final


def normalize_rule_check(rule_check: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(rule_check, dict):
        rule_check = {}

    decision = clean_text(rule_check.get("decision")).upper()
    if decision not in VALID_DECISIONS:
        decision = "NSTP"

    confidence = clean_text(rule_check.get("confidence")).upper()
    if confidence not in ["HIGH", "MEDIUM", "LOW"]:
        confidence = "LOW"

    violated_rules = dedupe_rules(rule_check.get("violated_rules", []))

    if decision == "STP":
        violated_rules = []

    return {
        **rule_check,
        "decision": decision,
        "stp_eligible": decision == "STP",
        "confidence": confidence,
        "violated_rules": violated_rules,
    }


def build_gui_summary(
    decision: str,
    violated_rules: List[Dict[str, Any]],
    required_documents: List[str],
    required_medical_reports: List[str],
    process_status: Dict[str, Any]
) -> Dict[str, Any]:
    if decision == "STP":
        return {
            "status_badge": "STP",
            "display_message": "Proposal can proceed through Straight-Through Processing.",
            "next_step": "No additional document request is suggested by the agent.",
            "highlight_items": []
        }

    highlight_items = []

    for rule in violated_rules:
        if isinstance(rule, dict):
            text = clean_text(rule.get("rule_id")) or clean_text(rule.get("rule_description"))
            if text:
                highlight_items.append(text)

    highlight_items.extend(required_documents)
    highlight_items.extend(required_medical_reports)

    if process_status.get("warnings"):
        highlight_items.extend(process_status.get("warnings", []))

    return {
        "status_badge": "NSTP",
        "display_message": "Proposal requires Non-Straight-Through Processing because one or more company rules appear to be violated.",
        "next_step": "Review violated rules, request suggested documents/reports, and pass the JSON to the Loading Proposal Agent.",
        "highlight_items": dedupe_text_list(highlight_items)
    }


def build_loading_agent_input(
    proposal_json: Dict[str, Any],
    decision: str,
    rule_check: Dict[str, Any],
    final_doc_analysis: Dict[str, Any],
    sql_context: Dict[str, Any],
    vector_context: Dict[str, Any]
) -> Dict[str, Any]:
    required_documents = dedupe_text_list(final_doc_analysis.get("required_documents", []))
    required_medical_reports = dedupe_text_list(final_doc_analysis.get("required_medical_reports", []))

    should_send = decision == "NSTP"

    return {
        "should_send_to_loading_agent": should_send,
        "reason": (
            "Proposal is NSTP after LLM checked the 78 embedded company rules."
            if should_send
            else "Proposal is STP; no loading handoff is required."
        ),
        "proposal_json": proposal_json,
        "proposal_summary": get_customer_summary(proposal_json),
        "stp_nstp_decision": decision,
        "violated_rules": rule_check.get("violated_rules", []),
        "required_documents": required_documents,
        "required_medical_reports": required_medical_reports,
        "document_recommendation_reasons": final_doc_analysis.get("document_recommendation_reasons", []),
        "possible_loading_types": final_doc_analysis.get("possible_loading_types", []),
        "retrieval_summary": {
            "sql_tool_used": bool(sql_context),
            "similar_sql_cases_count": len(sql_context.get("similar_cases", [])) if isinstance(sql_context, dict) else 0,
            "keyword_uw_remark_matches_count": len(sql_context.get("keyword_remark_matches", [])) if isinstance(sql_context, dict) else 0,
            "vector_docs_count": vector_context.get("final_docs_count", 0) if isinstance(vector_context, dict) else 0
        }
    }


def build_final_output_json(
    proposal_json: Dict[str, Any],
    rule_check: Dict[str, Any],
    retrieval_plan: Dict[str, Any],
    sql_context: Dict[str, Any],
    vector_context: Dict[str, Any],
    final_doc_analysis: Dict[str, Any],
    process_status: Dict[str, Any],
    status: str = "success",
    error: Optional[str] = None
) -> Dict[str, Any]:
    rule_check = normalize_rule_check(rule_check)

    decision = rule_check["decision"]
    violated_rules = rule_check["violated_rules"]

    required_documents = dedupe_text_list(final_doc_analysis.get("required_documents", []))
    required_medical_reports = dedupe_text_list(final_doc_analysis.get("required_medical_reports", []))
    document_reasons = dedupe_text_list(final_doc_analysis.get("document_recommendation_reasons", []))

    retrieval_summary = {
        "sql_tool_used": bool(sql_context),
        "vector_retrieval_used": bool(vector_context),
        "similar_sql_cases_count": len(sql_context.get("similar_cases", [])) if isinstance(sql_context, dict) else 0,
        "keyword_uw_remark_matches_count": len(sql_context.get("keyword_remark_matches", [])) if isinstance(sql_context, dict) else 0,
        "vector_docs_count": vector_context.get("final_docs_count", 0) if isinstance(vector_context, dict) else 0,
        "retrieval_plan": retrieval_plan,
    }

    output_json = {
        "proposalno": get_proposal_no(proposal_json),
        "customer_summary": get_customer_summary(proposal_json),

        "decision": decision,
        "stp_eligible": decision == "STP",
        "confidence": rule_check.get("confidence", "LOW"),

        "violated_rules": violated_rules,
        "rule_check_summary": {
            "rules_in_prompt_count": rule_check.get("rules_in_prompt_count"),
            "main_reasons": rule_check.get("main_reasons", []),
            "clean_stp_reason": rule_check.get("clean_stp_reason", ""),
            "ocr_verification_handling": rule_check.get("ocr_verification_handling", ""),
            "important_note": rule_check.get("important_note", "")
        },

        "required_documents": required_documents,
        "required_medical_reports": required_medical_reports,
        "document_recommendation_reasons": document_reasons,

        "retrieval_summary": retrieval_summary,
        "sql_past_case_context": sql_context,
        "vector_past_case_context": vector_context,

        "loading_agent_input": build_loading_agent_input(
            proposal_json=proposal_json,
            decision=decision,
            rule_check=rule_check,
            final_doc_analysis=final_doc_analysis,
            sql_context=sql_context,
            vector_context=vector_context
        ),

        "gui_summary": build_gui_summary(
            decision=decision,
            violated_rules=violated_rules,
            required_documents=required_documents,
            required_medical_reports=required_medical_reports,
            process_status=process_status
        ),

        "agent_process_status": process_status,
        "status": status,
        "error": error
    }

    return output_json
