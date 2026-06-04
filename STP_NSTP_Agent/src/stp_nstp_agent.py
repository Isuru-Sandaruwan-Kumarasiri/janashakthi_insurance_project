# =========================================================
# src/stp_nstp_agent.py
# LLM-first STP/NSTP Rule Checking + RAG Document Recommendation Agent
#
# Main public function:
# run_stp_nstp_agent(proposal_json: dict) -> dict
# =========================================================

from typing import Any, Dict, List
import json
import os

from langchain_openrouter import ChatOpenRouter

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    FINAL_OUTPUT_PATH,
    validate_runtime_files,
)
from src.schemas import (
    validate_proposal_input,
    proposal_input_to_dict,
    validate_agent_output,
    agent_output_to_dict,
    build_error_output,
)
from src.company_rules_prompt import (
    get_company_rules_prompt,
    get_company_rules_count,
    get_categorized_company_rules_prompt
)
from src.utils import (
    safe_json_dumps,
    extract_json_from_text,
    get_customer_summary,
    clean_text,
)
from src.sql_retriever import retrieve_sql_past_case_context
from src.vector_retriever import retrieve_vector_past_context
from src.output_builder import build_final_output_json, normalize_rule_check


# =========================================================
# LLM SETUP
# =========================================================

def create_openrouter_llm() -> ChatOpenRouter:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not loaded. "
            "Create .env with OPENROUTER_API_KEY."
        )

    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY.strip()

    return ChatOpenRouter(
        api_key=OPENROUTER_API_KEY.strip(),
        model=OPENROUTER_MODEL,
        temperature=1,
        max_retries=3,
        max_tokens=8000 # Added to prevent the JSON from getting cut off mid-generation
    )


def invoke_llm_json(
    llm: ChatOpenRouter,
    prompt: str,
    fallback: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        response = llm.invoke(prompt)
        parsed = extract_json_from_text(response.content)
        if parsed is None:
            fallback["llm_parse_error"] = "Model did not return valid JSON."
            fallback["raw_response_preview"] = str(response.content)[:1200]
            return fallback
        return parsed
    except Exception as e:
        fallback["llm_call_error"] = str(e)
        return fallback


# =========================================================
# PROMPTS
# =========================================================

def build_rule_check_prompt(
    proposal_json: Dict[str, Any],
    chat_history: List[Dict[str, str]] | None = None
) -> str:
    """
    Context-engineered prompt for direct company-rule checking.
    Uses categorized rules to isolate the LLM's attention block by block.
    """
    if chat_history is None:
        chat_history = []

    # Using the new categorized prompt to stop hallucination / crossover
    rules_prompt = get_categorized_company_rules_prompt()
    rules_count = get_company_rules_count()
    customer_summary = get_customer_summary(proposal_json)

    return f"""
You are a life insurance underwriting rule-checking agent.

To prevent misinterpretation and hallucination, the {rules_count} company rules are categorized into three parts: FINANCIAL, MEDICAL, and NON_MEDICAL.
You must evaluate the proposal systematically, section by section.

- When evaluating FINANCIAL rules, focus ONLY on `proposal_details` and `personal_details` (like monthly_income, sum_insured).
- When evaluating MEDICAL rules, focus ONLY on `medical_history`, `physical_details`, `family_history`, `habits`, and `female_questions`.
- When evaluating NON_MEDICAL rules, focus ONLY on `personal_details` (age, occupation), `previous_insurance_details`, `additional_questions`, and `customer_contact`.

You receive a verified proposal JSON from an OCR + Data Verification Agent.
That earlier OCR agent already checks whether the main customer details match ID/address proof:
- birth date / age
- NIC or ID number
- customer address
- main identity details
- missing proof or mismatch blocking checks

Therefore, in this agent:
- Do NOT fail the proposal using OCR mismatch rules unless the proposal JSON itself explicitly contains a mismatch/missing document.
- Do NOT invent document mismatches.
- Do NOT use data completeness score or authenticity score because this agent receives only verified proposals.

Your exact task:
1. Read the verified proposal JSON.
2. Check each categorized rule against the corresponding proposal data ONLY.
3. If at least one rule condition is truly violated, final decision must be "NSTP".
4. If no rule is violated, final decision must be "STP".
5. Only list rules as violated when the proposal data clearly satisfies that rule condition.
6. Some rules may require external policy history/core-system information. If the needed field is not present in the proposal JSON and no retrieved/external value is available at this stage, do not mark it as violated; mention it as "not checkable from current proposal data" only in important_note.
7. Do not retrieve rules from SQL or vector DB. All rules to check are already below.

Rules in prompt count: {rules_count}

VERIFIED PROPOSAL JSON:
{safe_json_dumps(proposal_json, max_chars=18000)}

CUSTOMER SUMMARY:
{safe_json_dumps(customer_summary, max_chars=4000)}

RECENT CHAT HISTORY, IF ANY:
{safe_json_dumps(chat_history[-6:], max_chars=2000)}

CATEGORIZED COMPANY STP/NSTP RULES:
{rules_prompt}

Return valid JSON only. No markdown. No extra text.

Use this exact JSON shape to strictly separate your logic:
{{
  "financial_evaluation": {{
    "violated_rules": [
      {{
        "rule_id": "example FINANCIAL_001",
        "rule_type": "FINANCIAL",
        "rule_description": "exact rule description",
        "condition_text": "condition from rule",
        "why_violated": "explain using exact proposal fields/values",
        "evidence_value": "proposal values that triggered this rule"
      }}
    ]
  }},
  "medical_evaluation": {{
    "violated_rules": []
  }},
  "non_medical_evaluation": {{
    "violated_rules": []
  }},
  "decision": "STP or NSTP",
  "stp_eligible": true,
  "confidence": "HIGH or MEDIUM or LOW",
  "rules_in_prompt_count": {rules_count},
  "main_reasons": [
    "short reason 1",
    "short reason 2"
  ],
  "clean_stp_reason": "If decision is STP, explain why no rule was violated. Otherwise empty string.",
  "ocr_verification_handling": "Explain that OCR mismatch rules were ignored only because OCR agent already verified and no mismatch was present, or explain actual mismatch if present.",
  "important_note": "Mention uncertainty or rules not checkable from current JSON, but do not mark unavailable rules as violated."
}}
""".strip()


def build_retrieval_plan_prompt(
    proposal_json: Dict[str, Any],
    rule_check: Dict[str, Any]
) -> str:
    """
    LLM decides how to use SQL and vector retrieval.
    The SQL tool never retrieves rules. Vector search filters rule docs.
    """
    return f"""
You are planning RAG retrieval for an insurance underwriting agent.

The STP/NSTP rule check is already completed using embedded company rules.
Now your job is only to plan retrieval of similar historical cases for document/report recommendation.

Available retrieval:
1. Custom SQL tool: sql_past_case_retriever
   - Retrieves similar proposal details, need analysis, medical details, questionnaire answers, and underwriter remarks.
   - Does NOT retrieve rules.
2. Vector retrieval:
   - Uses embedding similarity search over historical text/past evidence.
   - Must focus on similar past cases and document/report requests.
   - Must NOT search for company rules because rules are already in the prompt.

Create a retrieval plan using the proposal and rule-check result.

PROPOSAL JSON:
{safe_json_dumps(proposal_json, max_chars=10000)}

RULE CHECK RESULT:
{safe_json_dumps(rule_check, max_chars=8000)}

Return valid JSON only. No markdown.

Use this exact JSON shape:
{{
  "use_sql_tool": true,
  "sql_tool_name": "sql_past_case_retriever",
  "sql_tool_reason": "why SQL historical retrieval is useful",
  "sql_limit": 8,
  "use_vector_retrieval": true,
  "vector_query": "one strong semantic query for similar historical underwriting cases and required documents/reports",
  "vector_reason": "why this vector query is useful",
  "do_not_retrieve_rules_note": "confirm that rules must not be retrieved from SQL/vector"
}}
""".strip()


def build_document_recommendation_prompt(
    proposal_json: Dict[str, Any],
    rule_check: Dict[str, Any],
    retrieval_plan: Dict[str, Any],
    sql_context: Dict[str, Any],
    vector_context: Dict[str, Any]
) -> str:
    """
    Final LLM prompt to recommend documents and reports based on
    violated rules + similar past evidence.
    """
    return f"""
You are an insurance underwriting document recommendation agent.

The STP/NSTP decision is already made from the embedded company-rule check.
Your task now:
1. Use the violated rules, SQL historical cases, vector evidence, and underwriter remarks.
2. Decide what documents or medical reports should be requested.
3. Use similar past cases only as supporting evidence.
4. Do NOT create new STP/NSTP rules.
5. Do NOT change STP to NSTP or NSTP to STP.
6. If decision is STP, return empty required_documents and required_medical_reports unless a clear non-rule operational requirement is explicitly shown.
7. If decision is NSTP, recommend practical documents/reports that should go to the customer or next Loading Proposal Agent.
8. If past evidence mentions a medical condition such as diabetes, hypertension, asthma, smoking, alcohol, hazardous occupation, etc., recommend suitable evidence only when it matches the new proposal.

RULE CHECK RESULT:
{safe_json_dumps(rule_check, max_chars=9000)}

RETRIEVAL PLAN:
{safe_json_dumps(retrieval_plan, max_chars=3000)}

SQL HISTORICAL CONTEXT:
{safe_json_dumps(sql_context, max_chars=16000)}

VECTOR HISTORICAL CONTEXT:
{safe_json_dumps(vector_context, max_chars=12000)}

PROPOSAL JSON:
{safe_json_dumps(proposal_json, max_chars=10000)}

Return valid JSON only. No markdown.

Use this exact JSON shape:
{{
  "required_documents": [
    "document 1"
  ],
  "required_medical_reports": [
    "medical report 1"
  ],
  "document_recommendation_reasons": [
    "reason based on rule or similar past case"
  ],
  "possible_loading_types": [
    "HEALTH_LOADING or LIFESTYLE_LOADING or OCCUPATION_LOADING or FINANCIAL_LOADING if relevant"
  ],
  "evidence_used": [
    {{
      "source": "sql or vector or violated_rule",
      "reference": "short reference",
      "how_it_helped": "explanation"
    }}
  ],
  "next_agent_instruction": "short instruction for Loading Proposal Agent"
}}
""".strip()


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_retrieval_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        plan = {}

    vector_query = clean_text(plan.get("vector_query"))
    if not vector_query:
        vector_query = "similar historical underwriting cases required documents medical reports underwriter remarks"

    try:
        sql_limit = int(plan.get("sql_limit", 8))
    except Exception:
        sql_limit = 8

    sql_limit = max(1, min(sql_limit, 20))

    return {
        "use_sql_tool": bool(plan.get("use_sql_tool", True)),
        "sql_tool_name": "sql_past_case_retriever",
        "sql_tool_reason": clean_text(plan.get("sql_tool_reason")) or "Retrieve similar historical cases and underwriter remarks.",
        "sql_limit": sql_limit,
        "use_vector_retrieval": bool(plan.get("use_vector_retrieval", True)),
        "vector_query": vector_query,
        "vector_reason": clean_text(plan.get("vector_reason")) or "Retrieve similar unstructured historical evidence.",
        "do_not_retrieve_rules_note": clean_text(plan.get("do_not_retrieve_rules_note")) or "Rules are embedded in prompt and are not retrieved."
    }


def normalize_document_analysis(
    final_doc_analysis: Dict[str, Any],
    decision: str
) -> Dict[str, Any]:
    if not isinstance(final_doc_analysis, dict):
        final_doc_analysis = {}

    for key in [
        "required_documents",
        "required_medical_reports",
        "document_recommendation_reasons",
        "possible_loading_types",
        "evidence_used"
    ]:
        if not isinstance(final_doc_analysis.get(key), list):
            final_doc_analysis[key] = []

    if decision == "STP":
        final_doc_analysis["required_documents"] = []
        final_doc_analysis["required_medical_reports"] = []
        if not final_doc_analysis["document_recommendation_reasons"]:
            final_doc_analysis["document_recommendation_reasons"] = [
                "No company rule was violated, so no additional document or medical report is required at this stage."
            ]
        final_doc_analysis["possible_loading_types"] = []

    if not clean_text(final_doc_analysis.get("next_agent_instruction")):
        final_doc_analysis["next_agent_instruction"] = (
            "Use violated rules and required evidence if this proposal is sent to the Loading Proposal Agent."
        )

    return final_doc_analysis


# =========================================================
# MAIN AGENT FUNCTION
# =========================================================

def run_stp_nstp_agent(
    proposal_json: Dict[str, Any],
    chat_history: List[Dict[str, str]] | None = None,
    save_output: bool = True,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Run the LLM-first STP/NSTP agent.

    Input:
        proposal_json: verified proposal JSON from OCR + Verification Agent.

    Output:
        JSON for GUI and Loading Proposal Agent.
    """
    try:
        validate_runtime_files(strict=True)

        validated_input = validate_proposal_input(proposal_json)
        proposal_json = proposal_input_to_dict(validated_input)

        if chat_history is None:
            chat_history = [
                {
                    "role": "system",
                    "content": "OCR + Verification Agent has already verified identity, address, NIC/DOB, and mismatch checks."
                },
                {
                    "role": "user",
                    "content": "Check embedded company rules, decide STP/NSTP, retrieve similar past cases, and recommend required documents."
                }
            ]

        process_status = {
            "rule_check_completed": False,
            "retrieval_plan_completed": False,
            "sql_retrieval_completed": False,
            "vector_retrieval_completed": False,
            "final_document_analysis_completed": False,
            "warnings": [],
            "architecture_note": (
                "LLM checks all embedded company rules section-by-section in the prompt to avoid hallucination. "
                "SQL and vector retrieval are used only for similar past-case evidence and document recommendation."
            )
        }

        if not use_llm:
            raise RuntimeError(
                "This new version is LLM-first. Set use_llm=True because rule checking is done in the prompt."
            )

        llm = create_openrouter_llm()

        # -------------------------------------------------
        # 1. Rule check using categorized sections in prompt
        # -------------------------------------------------
        rule_check_prompt = build_rule_check_prompt(
            proposal_json=proposal_json,
            chat_history=chat_history
        )

        rule_check_fallback = {
            "decision": "NSTP",
            "stp_eligible": False,
            "confidence": "LOW",
            "rules_in_prompt_count": get_company_rules_count(),
            "main_reasons": [
                "LLM rule check failed or returned invalid JSON. Proposal routed to NSTP for safety."
            ],
            "financial_evaluation": {"violated_rules": []},
            "medical_evaluation": {"violated_rules": []},
            "non_medical_evaluation": {"violated_rules": []},
            "clean_stp_reason": "",
            "ocr_verification_handling": "OCR verification was completed before this agent.",
            "important_note": "Fallback output because rule-check LLM response was invalid."
        }

        rule_check = invoke_llm_json(
            llm=llm,
            prompt=rule_check_prompt,
            fallback=rule_check_fallback
        )

        # Captures hidden LLM errors to show in the GUI
        if "llm_call_error" in rule_check:
            process_status["llm_call_error"] = rule_check["llm_call_error"]
            process_status["warnings"].append("Rule check LLM API call failed.")
        elif "llm_parse_error" in rule_check:
            process_status["llm_parse_error"] = rule_check["llm_parse_error"]
            process_status["raw_response_preview"] = rule_check.get("raw_response_preview", "")
            process_status["warnings"].append("Rule check LLM returned invalid JSON (likely cut off due to max_tokens limit).")

        # -- NEW: Extract categorized violated rules into a flat list --
        # This keeps app.py, output_builder.py, and schemas.py perfectly compatible
        flat_violated_rules = []
        if isinstance(rule_check.get("financial_evaluation"), dict):
            flat_violated_rules.extend(rule_check["financial_evaluation"].get("violated_rules", []))
        if isinstance(rule_check.get("medical_evaluation"), dict):
            flat_violated_rules.extend(rule_check["medical_evaluation"].get("violated_rules", []))
        if isinstance(rule_check.get("non_medical_evaluation"), dict):
            flat_violated_rules.extend(rule_check["non_medical_evaluation"].get("violated_rules", []))

        # Catch if LLM put them in root by mistake anyway
        if isinstance(rule_check.get("violated_rules"), list):
            flat_violated_rules.extend(rule_check["violated_rules"])

        rule_check["violated_rules"] = flat_violated_rules

        rule_check = normalize_rule_check(rule_check)
        process_status["rule_check_completed"] = True

        # -------------------------------------------------
        # 2. LLM retrieval plan
        # -------------------------------------------------
        retrieval_plan_prompt = build_retrieval_plan_prompt(
            proposal_json=proposal_json,
            rule_check=rule_check
        )

        retrieval_plan_fallback = {
            "use_sql_tool": True,
            "sql_tool_name": "sql_past_case_retriever",
            "sql_tool_reason": "Retrieve similar historical cases and underwriter remarks.",
            "sql_limit": 8,
            "use_vector_retrieval": True,
            "vector_query": "similar historical underwriting cases required documents medical reports underwriter remarks",
            "vector_reason": "Retrieve similar unstructured past evidence.",
            "do_not_retrieve_rules_note": "Rules are embedded in prompt and are not retrieved from SQL/vector."
        }

        retrieval_plan = invoke_llm_json(
            llm=llm,
            prompt=retrieval_plan_prompt,
            fallback=retrieval_plan_fallback
        )

        # Captures retrieval plan LLM errors
        if "llm_call_error" in retrieval_plan:
            process_status["warnings"].append(f"Retrieval Plan LLM API error: {retrieval_plan['llm_call_error']}")
        elif "llm_parse_error" in retrieval_plan:
            process_status["warnings"].append("Retrieval Plan LLM returned invalid JSON.")

        retrieval_plan = normalize_retrieval_plan(retrieval_plan)
        process_status["retrieval_plan_completed"] = True

        # -------------------------------------------------
        # 3. SQL tool retrieval, without rules
        # -------------------------------------------------
        sql_context = {}
        if retrieval_plan["use_sql_tool"]:
            try:
                sql_context = retrieve_sql_past_case_context(
                    proposal_json=proposal_json,
                    rule_check=rule_check,
                    limit=retrieval_plan["sql_limit"]
                )
                process_status["sql_retrieval_completed"] = True
            except Exception as sql_error:
                sql_context = {
                    "tool_name": "sql_past_case_retriever",
                    "status": "error",
                    "error": str(sql_error),
                    "similar_cases": [],
                    "keyword_remark_matches": []
                }
                process_status["warnings"].append(f"SQL retrieval failed: {sql_error}")

        # -------------------------------------------------
        # 4. Vector retrieval using LLM-generated query
        # -------------------------------------------------
        vector_context = {}
        if retrieval_plan["use_vector_retrieval"]:
            try:
                vector_context = retrieve_vector_past_context(
                    vector_query=retrieval_plan["vector_query"],
                    k_initial=20,
                    k_final=8
                )
                process_status["vector_retrieval_completed"] = True
            except Exception as vector_error:
                vector_context = {
                    "status": "error",
                    "error": str(vector_error),
                    "vector_query": retrieval_plan["vector_query"],
                    "final_docs_count": 0,
                    "candidate_vector_docs": []
                }
                process_status["warnings"].append(f"Vector retrieval failed: {vector_error}")

        # -------------------------------------------------
        # 5. Final document/report recommendation
        # -------------------------------------------------
        doc_prompt = build_document_recommendation_prompt(
            proposal_json=proposal_json,
            rule_check=rule_check,
            retrieval_plan=retrieval_plan,
            sql_context=sql_context,
            vector_context=vector_context
        )

        doc_fallback = {
            "required_documents": [],
            "required_medical_reports": [],
            "document_recommendation_reasons": [
                "Document recommendation LLM failed or returned invalid JSON."
            ],
            "possible_loading_types": [],
            "evidence_used": [],
            "next_agent_instruction": "Review proposal manually because document recommendation failed."
        }

        final_doc_analysis = invoke_llm_json(
            llm=llm,
            prompt=doc_prompt,
            fallback=doc_fallback
        )

        # Captures doc analysis LLM errors
        if "llm_call_error" in final_doc_analysis:
            process_status["warnings"].append(f"Doc Analysis LLM API error: {final_doc_analysis['llm_call_error']}")
        elif "llm_parse_error" in final_doc_analysis:
            process_status["warnings"].append("Doc Analysis LLM returned invalid JSON.")

        final_doc_analysis = normalize_document_analysis(
            final_doc_analysis=final_doc_analysis,
            decision=rule_check["decision"]
        )
        process_status["final_document_analysis_completed"] = True

        # -------------------------------------------------
        # 6. Build final output
        # -------------------------------------------------
        output_json = build_final_output_json(
            proposal_json=proposal_json,
            rule_check=rule_check,
            retrieval_plan=retrieval_plan,
            sql_context=sql_context,
            vector_context=vector_context,
            final_doc_analysis=final_doc_analysis,
            process_status=process_status,
            status="success",
            error=None
        )

        validated_output = validate_agent_output(output_json)
        output_json = agent_output_to_dict(validated_output)

        if save_output:
            FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)

        return output_json

    except Exception as e:
        error_output = build_error_output(
            proposal_json=proposal_json,
            error_message=str(e)
        )

        if save_output:
            FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(error_output, f, indent=4, ensure_ascii=False)

        return error_output