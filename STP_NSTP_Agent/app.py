# =========================================================
# app.py
# Streamlit GUI for LLM-first STP/NSTP Rule RAG Agent
#
# Important:
# - OCR/document upload is NOT handled here.
# - This GUI simulates the verified JSON received after OCR + Verification Agent.
# - No data completeness/authenticity score is collected.
# =========================================================

from pathlib import Path
import sys
import json
import datetime as dt
import html
from typing import Any, Dict, List

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_runtime_status, validate_runtime_files, OPENROUTER_MODEL
from src.stp_nstp_agent import run_stp_nstp_agent


st.set_page_config(
    page_title="LLM Rule RAG STP/NSTP Agent",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# CSS - force clean light UI even if Streamlit/browser is dark
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #f6f8fb;
        --panel: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --border: #cbd5e1;
        --border-soft: #e2e8f0;
        --primary: #1f4e79;
        --primary-hover: #163b5c;
        --success-bg: #ecfdf3;
        --success-border: #86efac;
        --warning-bg: #fff7ed;
        --warning-border: #fdba74;
        --error-bg: #fef2f2;
        --error-border: #fca5a5;
        --info-bg: #eff6ff;
        --info-border: #bfdbfe;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp > header,
    header[data-testid="stHeader"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 1260px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--border-soft) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text) !important;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: var(--text) !important;
    }

    .sub-title {
        font-size: 16px;
        color: var(--muted) !important;
        margin-bottom: 1.5rem;
    }

    .good-box,
    .warn-box,
    .bad-box,
    .info-box,
    .white-card {
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 14px;
        color: var(--text) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        line-height: 1.5;
    }

    .good-box {
        background: var(--success-bg) !important;
        border: 1px solid var(--success-border) !important;
    }

    .warn-box {
        background: var(--warning-bg) !important;
        border: 1px solid var(--warning-border) !important;
    }

    .bad-box {
        background: var(--error-bg) !important;
        border: 1px solid var(--error-border) !important;
    }

    .info-box {
        background: var(--info-bg) !important;
        border: 1px solid var(--info-border) !important;
    }

    .white-card {
        background: #ffffff !important;
        border: 1px solid var(--border-soft) !important;
    }

    .small-note {
        color: var(--muted) !important;
        font-size: 13px;
    }

    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        margin-left: 6px;
    }

    .status-true {
        background: #dcfce7 !important;
        color: #166534 !important;
        border: 1px solid #86efac !important;
    }

    .status-false {
        background: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fca5a5 !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] button:focus {
        background: #f8fafc !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: var(--text) !important;
        stroke: var(--text) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    div[data-baseweb="select"] * {
        color: var(--text) !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--text) !important;
    }

    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"],
    div[role="option"] {
        background-color: #ffffff !important;
        color: var(--text) !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background-color: #eaf2fb !important;
        color: var(--text) !important;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button,
    .stDownloadButton > button {
        background: var(--primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--primary) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    .stButton > button *,
    [data-testid="stFormSubmitButton"] button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    .stDownloadButton > button:hover {
        background: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
        color: #ffffff !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
        padding: 14px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetric"] * {
        color: var(--text) !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] * {
        color: var(--text) !important;
    }

    .stAlert {
        background: var(--info-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--info-border) !important;
        border-radius: 12px !important;
    }

    .stAlert * {
        color: var(--text) !important;
    }

    .clean-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .clean-table th {
        background: #f1f5f9;
        color: var(--text);
        text-align: left;
        padding: 12px;
        border-bottom: 1px solid var(--border-soft);
        font-weight: 700;
    }

    .clean-table td {
        background: #ffffff;
        color: var(--text);
        padding: 12px;
        border-bottom: 1px solid var(--border-soft);
        vertical-align: top;
    }

    .clean-table tr:last-child td {
        border-bottom: none;
    }

    code, pre {
        background: #f1f5f9 !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return html.escape(json.dumps(value, ensure_ascii=False))
    return html.escape(str(value))


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def yn_select(label: str, default: bool = False, key: str | None = None) -> bool:
    value = st.selectbox(label, ["No", "Yes"], index=1 if default else 0, key=key)
    return value == "Yes"


def json_text(obj: Any) -> str:
    return json.dumps(obj or {}, indent=4, ensure_ascii=False)


def safe_list(items: Any) -> List[Any]:
    return items if isinstance(items, list) else []


def render_list(items: Any, empty_message: str, success_empty: bool = False) -> None:
    items = safe_list(items)
    if items:
        for item in items:
            st.write("- " + str(item))
    else:
        if success_empty:
            st.success(empty_message)
        else:
            st.info(empty_message)


def render_clean_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No records to display.")
        return

    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(col)}</th>" for col in columns)

    body = ""
    for row in rows:
        body += "<tr>"
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)
            elif isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)
            body += f"<td>{escape(value)}</td>"
        body += "</tr>"

    st.markdown(
        f"""
        <table class="clean-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{body}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def bool_badge(value: bool) -> str:
    css_class = "status-true" if bool(value) else "status-false"
    label = "True" if bool(value) else "False"
    return f'<span class="status-pill {css_class}">{label}</span>'


def sidebar_status(label: str, value: bool) -> None:
    st.markdown(f"<b>{escape(label)}:</b> {bool_badge(value)}", unsafe_allow_html=True)


def render_rule_cards(rules: Any) -> None:
    rules = safe_list(rules)
    if not rules:
        st.success("No violated company rule was returned.")
        return

    for i, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            continue
        st.markdown(
            f"""
            <div class="warn-box">
                <b>{escape(rule.get("rule_id") or f"Rule {i}")}</b><br>
                <span class="small-note">Type: {escape(rule.get("rule_type", ""))}</span>
                <p><b>Description:</b> {escape(rule.get("rule_description", ""))}</p>
                <p><b>Condition:</b> {escape(rule.get("condition_text", ""))}</p>
                <p><b>Why violated:</b> {escape(rule.get("why_violated", ""))}</p>
                <p><b>Evidence value:</b> {escape(rule.get("evidence_value", ""))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_proposal_json(form: dict) -> dict:
    proposalno = form["proposalno"].strip()
    if proposalno == "":
        proposalno = "FORM-PROP-" + dt.datetime.now().strftime("%Y%m%d%H%M%S")

    return {
        "proposal_metadata": {
            "proposalno": proposalno,
            "policy_no": None,
            "source": "ocr_verified_frontend_form",
            "form_type": "Janashakthi Proposal for Life Insurance",
        },
        "customer_contact": {
            "full_name": form["full_name"],
            "name_with_initials": form["name_with_initials"],
            "telephone_land": None,
            "mobile": form["mobile"],
            "email": form["email"],
            "preferred_language": form["preferred_language"],
        },
        "personal_details": {
            "age": form["age"],
            "date_of_birth": form["date_of_birth"],
            "nic": "MASKED_VERIFIED_NIC",
            "gender": form["gender"],
            "marital_status": form["marital_status"],
            "address": form["address"],
            "occupation": form["occupation"],
            "company_or_institution": form["company_or_institution"],
            "monthly_income": form["monthly_income"],
        },
        "proposal_details": {
            "plan_type": form["plan_type"],
            "sum_insured": str(form["sum_insured"]),
            "premium_amount": form["premium_amount"],
            "payment_frequency": form["payment_frequency"],
            "payment_method": form["payment_method"],
            "riders": form["riders"],
        },
        "previous_insurance_details": {
            "has_existing_or_previous_policy": yes_no(form["has_existing_or_previous_policy"]),
            "has_declined_or_postponed_policy": yes_no(form["has_declined_or_postponed_policy"]),
            "details": [],
        },
        "physical_details": {
            "height_cm": form["height_cm"],
            "weight_kg": form["weight_kg"],
            "bmi": None,
        },
        "habits": {
            "smoker": yes_no(form["smoker"]),
            "smoking_quantity_per_week": None,
            "smoking_duration_years": None,
            "alcohol": yes_no(form["alcohol"]),
            "alcohol_quantity_ml": None,
            "alcohol_duration_years": None,
        },
        "family_history": {
            "has_family_medical_history": yes_no(form["has_family_medical_history"]),
            "family_disease_details": [],
        },
        "medical_history": {
            "visited_doctor_last_3_years": yes_no(form["visited_doctor_last_3_years"]),
            "had_medical_condition_or_injury": yes_no(form["had_medical_condition_or_injury"]),
            "had_operation_xray_or_hospital_test": yes_no(form["had_operation_xray_or_hospital_test"]),
            "overnight_hospital_stay": yes_no(form["overnight_hospital_stay"]),
            "currently_taking_treatment_or_drugs": yes_no(form["currently_taking_treatment_or_drugs"]),
            "mental_or_neurological_condition": yes_no(form["mental_or_neurological_condition"]),
            "physical_disability_or_defect": yes_no(form["physical_disability_or_defect"]),
            "physical_disability_deformity_or_impairment": yes_no(form["physical_disability_or_defect"]),
            "absent_from_work_due_to_health": yes_no(form["absent_from_work_due_to_health"]),
            "specific_disease_history": {
                "heart_or_blood_pressure_or_stroke": yes_no(form["heart_or_blood_pressure_or_stroke"]),
                "kidney_or_urinary_or_eyes_ears": yes_no(form["kidney_or_urinary_or_eyes_ears"]),
                "brain_or_nervous_or_mental": yes_no(form["brain_or_nervous_or_mental"]),
                "cancer_or_tumour": yes_no(form["cancer_or_tumour"]),
                "multiple_sclerosis_or_arthritis_or_rheumatism": yes_no(form["multiple_sclerosis_or_arthritis_or_rheumatism"]),
                "hepatitis_aids_or_related_condition": yes_no(form["hepatitis_aids_or_related_condition"]),
                "respiratory_or_asthma_or_bronchitis": yes_no(form["respiratory_or_asthma_or_bronchitis"]),
                "digestive_gall_bladder_liver_ulcer_bleeding": yes_no(form["digestive_gall_bladder_liver_ulcer_bleeding"]),
            },
            "medical_details_text": form["medical_details_text"],
        },
        "additional_questions": {
            "hazardous_occupation": yes_no(form["hazardous_occupation"]),
            "hazardous_sport": yes_no(form["hazardous_sport"]),
            "criminal_offence": yes_no(form["criminal_offence"]),
            "threat_on_life": yes_no(form["threat_on_life"]),
        },
        "female_questions": {
            "pregnant_present": yes_no(form["pregnant_present"]) if form["gender"] == "Female" else None,
            "pregnancy_complication_history": yes_no(form["pregnancy_complication_history"]) if form["gender"] == "Female" else None,
            "female_organ_disorder": yes_no(form["female_organ_disorder"]) if form["gender"] == "Female" else None,
        },
        "branch_office_checks": {
            "q11_family_question_yes": yes_no(form["q11_family_question_yes"]),
            "q12_medical_question_yes": yes_no(form["q12_medical_question_yes"]),
            "q13_additional_question_yes": yes_no(form["q13_additional_question_yes"]),
            "q14_female_question_yes": yes_no(form["q14_female_question_yes"]),
        },
        "verification_result": {
            "identity_verified": True,
            "address_verified": True,
            "nic_verified": True,
            "dob_verified": True,
            "mismatches": [],
            "missing_documents": [],
            "note": "OCR + Verification Agent already verified these fields before this agent received the JSON.",
        },
    }


def render_loading_agent_section(output: Dict[str, Any]) -> None:
    loading_input = output.get("loading_agent_input", {}) or {}
    should_send = bool(loading_input.get("should_send_to_loading_agent", False))
    reason = loading_input.get("reason", "")
    possible_loading_types = safe_list(loading_input.get("possible_loading_types", []))

    st.markdown("### Loading Agent Handoff")
    box = "warn-box" if should_send else "good-box"
    title = "Send to Loading Agent: Yes" if should_send else "Send to Loading Agent: No"
    st.markdown(
        f"""
        <div class="{box}">
            <b>{escape(title)}</b><br>
            {escape(reason)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Handoff", "Yes" if should_send else "No")
    c2.metric("Violated Rules", len(safe_list(loading_input.get("violated_rules", []))))
    c3.metric("Loading Types", len(possible_loading_types))

    st.write("**Possible loading types:**")
    render_list(possible_loading_types, "No loading type detected.", success_empty=True)

    with st.expander("View final JSON passed to Loading Agent"):
        loading_json_text = json_text(loading_input)
        st.text_area(
            "Copyable Loading Agent JSON",
            value=loading_json_text,
            height=420,
            help="Click inside this box, press Ctrl+A, then Ctrl+C.",
        )
        st.download_button(
            label="Download Loading Agent JSON",
            data=loading_json_text,
            file_name="loading_agent_input.json",
            mime="application/json",
            use_container_width=True,
        )


def render_retrieval_summary(output: Dict[str, Any]) -> None:
    summary = output.get("retrieval_summary", {}) or {}
    st.markdown("### Retrieval Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SQL Tool Used", str(summary.get("sql_tool_used", False)))
    c2.metric("Similar SQL Cases", summary.get("similar_sql_cases_count", 0))
    c3.metric("UW Remark Matches", summary.get("keyword_uw_remark_matches_count", 0))
    c4.metric("Vector Docs", summary.get("vector_docs_count", 0))

    plan = summary.get("retrieval_plan", {}) or {}
    if plan:
        render_clean_table([
            {
                "Use SQL": plan.get("use_sql_tool"),
                "Use Vector": plan.get("use_vector_retrieval"),
                "SQL Limit": plan.get("sql_limit"),
                "Vector Query": plan.get("vector_query"),
            }
        ])

    
    with st.expander("🗄️ SQL Past-Case Context", expanded=False):
        sql_data = output.get("sql_past_case_context", {})
        
        if sql_data.get("status") == "error":
            st.error(f"SQL Retrieval Error: {sql_data.get('error')}")
        elif not sql_data:
            st.info("No SQL context available.")
        else:
            # Display Keywords
            keywords = sql_data.get("retrieval_keywords", [])
            if keywords:
                st.markdown(f"**Retrieval Keywords:** `{', '.join(keywords)}`")
            
            st.markdown("---")
            st.markdown("#### Similar Base Cases")
            similar_cases = sql_data.get("similar_cases", [])
            if similar_cases:
                # Flatten the base cases for a clean dataframe view
                flat_cases = [case.get("base_case", {}) for case in similar_cases]
                st.dataframe(flat_cases, use_container_width=True)
            else:
                st.info("No similar base cases found in historical data.")

            st.markdown("---")
            st.markdown("#### Underwriter Remark Matches")
            remarks = sql_data.get("keyword_remark_matches", [])
            if remarks:
                st.dataframe(remarks, use_container_width=True)
            else:
                st.info("No historical underwriter remarks matched the keywords.")

    with st.expander("📄 Vector Past-Case Context", expanded=False):
        vector_data = output.get("vector_past_case_context", {})
        
        if vector_data.get("status") == "error":
            st.error(f"Vector Retrieval Error: {vector_data.get('error')}")
        elif not vector_data:
            st.info("No Vector context available.")
        else:
            # Display the LLM-generated vector query
            query = vector_data.get('vector_query', 'N/A')
            st.markdown(f"**Vector Query:** _{query}_")
            
            # Display doc count
            st.metric("Documents Retrieved", vector_data.get("final_docs_count", 0))
            st.markdown("---")

            # Display actual retrieved documents
            docs = vector_data.get("candidate_vector_docs", [])
            if docs:
                for i, doc in enumerate(docs):
                    with st.expander(f"Matched Document {i+1}"):
                        # --- NEW: Bulletproof extraction logic ---
                        if isinstance(doc, dict):
                            # Try common dictionary keys for serialized docs
                            content = doc.get("page_content") or doc.get("content") or doc.get("text") or "No text content found in document dictionary."
                            metadata = doc.get("metadata", {})
                        else:
                            # Fallback just in case it is still a raw Langchain Document object
                            content = getattr(doc, "page_content", "No text content found in Document object.")
                            metadata = getattr(doc, "metadata", {})
                        # -----------------------------------------
                        
                        st.write(content)
                        st.caption(f"Metadata: {metadata}")
            else:
                st.info("No relevant documents found in the vector store.")


def render_output(output: Dict[str, Any]) -> None:
    decision = output.get("decision", "NSTP")
    status = output.get("status", "success")
    gui_summary = output.get("gui_summary", {}) or {}
    box = "good-box" if decision == "STP" else "warn-box"
    if status == "error":
        box = "bad-box"

    st.markdown("## Agent Output")
    st.markdown(
        f"""
        <div class="{box}">
            <h2>{escape(decision)}</h2>
            <p><b>{escape(gui_summary.get("display_message", ""))}</b></p>
            <p>{escape(gui_summary.get("next_step", ""))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", decision)
    c2.metric("STP Eligible", "Yes" if output.get("stp_eligible") else "No")
    c3.metric("Confidence", output.get("confidence", ""))
    c4.metric("Status", status)

    if output.get("error"):
        st.error(output["error"])

    # ---------------------------------------------------------
    # NEW BLOCK: STP Hidden Risk Assessment
    # ---------------------------------------------------------
    if decision == "STP":
        risk_level = output.get("stp_risk_level", "LOW")
        risk_box = "good-box" if risk_level == "LOW" else ("warn-box" if risk_level == "MEDIUM" else "bad-box")
        
        st.markdown("### Historical Risk Assessment (STP Cases)")
        st.markdown(
            f"""
            <div class="{risk_box}">
                <b>Hidden Risk Level: {escape(risk_level)}</b><br>
                This assessment compares the current STP proposal against similar past cases.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_list(output.get("stp_risk_factors", []), "No historical risk factors found.", success_empty=True)
    # ---------------------------------------------------------

    customer = output.get("customer_summary", {}) or {}
    st.markdown("### Customer / Proposal Snapshot")
    render_clean_table([
        {
            "Proposal No": customer.get("proposalno"),
            "Customer Name": customer.get("full_name"),
            "Age": customer.get("age"),
            "Gender": customer.get("gender"),
            "Marital Status": customer.get("marital_status"),
            "Occupation": customer.get("occupation"),
            "Monthly Income": customer.get("monthly_income"),
            "Sum Insured": customer.get("sum_insured"),
            "Plan Type": customer.get("plan_type"),
        }
    ])

    rule_summary = output.get("rule_check_summary", {}) or {}
    st.markdown("### Rule Check Summary")
    render_clean_table([
        {
            "Rules in Prompt": rule_summary.get("rules_in_prompt_count"),
            "Main Reasons": " | ".join(str(x) for x in safe_list(rule_summary.get("main_reasons", []))),
            "Clean STP Reason": rule_summary.get("clean_stp_reason", ""),
            "OCR Verification Handling": rule_summary.get("ocr_verification_handling", ""),
            "Important Note": rule_summary.get("important_note", ""),
        }
    ])

    st.markdown("### Violated Rules")
    render_rule_cards(output.get("violated_rules", []))

    left, right = st.columns(2)
    with left:
        st.markdown("### Required Documents")
        render_list(output.get("required_documents", []), "No additional documents required.", success_empty=True)
    with right:
        st.markdown("### Required Medical Reports")
        render_list(output.get("required_medical_reports", []), "No medical reports required.", success_empty=True)

    st.markdown("### Document Recommendation Reasons")
    render_list(
        output.get("document_recommendation_reasons", []),
        "No document recommendation reasons returned.",
        success_empty=False,
    )

    render_loading_agent_section(output)
    render_retrieval_summary(output)

    with st.expander("⚙️ Agent Process Status", expanded=True):
        status_data = output.get("agent_process_status", {})
        
        # Define the steps based on your stp_nstp_agent.py schema
        steps = [
            ("Rule Check", "rule_check_completed"),
            ("Retrieval Plan", "retrieval_plan_completed"),
            ("SQL Retrieval", "sql_retrieval_completed"),
            ("Vector Retrieval", "vector_retrieval_completed"),
            ("Final Document Analysis", "final_document_analysis_completed")
        ]
        
        # Display step status
        for label, key in steps:
            if status_data.get(key):
                st.markdown(f"✅ **{label}**: Completed")
            else:
                st.markdown(f"⏳ **{label}**: Pending or Skipped")

        # Display internal LLM Debug Info if it failed
        if "llm_call_error" in status_data:
            st.markdown("---")
            st.error(f"🛑 **LLM API Error:**\n{status_data['llm_call_error']}")
            
        if "llm_parse_error" in status_data:
            st.markdown("---")
            st.error(f"⚠️ **LLM Parsing Error:**\n{status_data['llm_parse_error']}")
            if status_data.get("raw_response_preview"):
                st.markdown("**Raw Output Preview from LLM:** *(Notice if the JSON brackets get cut off at the end)*")
                st.code(status_data["raw_response_preview"], language="json")

        # Display warnings if any
        warnings = status_data.get("warnings", [])
        if warnings:
            st.markdown("---")
            for w in warnings:
                st.warning(w)

        # Display the architecture note
        note = status_data.get("architecture_note")
        if note:
            st.info(note)
   

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">LLM Rule RAG STP/NSTP Underwriting Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Verified proposal JSON → 78 embedded company rules → SQL/vector past evidence → document recommendation → Loading Agent JSON</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Runtime")
    use_llm = True
    save_output = st.selectbox("Save final output JSON", ["Yes", "No"], index=0) == "Yes"

    status = get_runtime_status()
    st.write("Project dir:", status.get("project_dir", ""))
    sidebar_status("DB exists", status.get("database_exists", False))
    sidebar_status("Vector store folder exists", status.get("vector_store_dir_exists", False))
    st.write("Detected Chroma folders:", status.get("detected_chroma_count", 0))
    sidebar_status("OpenRouter key loaded", status.get("openrouter_api_key_loaded", False))
    st.write("OpenRouter model:", status.get("openrouter_model", OPENROUTER_MODEL))
    st.write("Embedding model:", status.get("embedding_model", ""))

    if st.button("Validate Runtime Files"):
        validation = validate_runtime_files(strict=False)
        if validation.get("ready"):
            st.success("Runtime is ready.")
        else:
            st.warning("Missing items:")
            for item in validation.get("missing_items", []):
                st.write("- " + str(item))


# =========================================================
# INPUT
# =========================================================
input_method = st.selectbox(
    "Input method",
    ["Fill verified proposal form", "Paste verified proposal JSON"],
)

proposal_json = None


if input_method == "Paste verified proposal JSON":
    st.markdown("## Paste Verified Proposal JSON")
    raw_json = st.text_area("Proposal JSON", height=430)

    if st.button("Run Agent", use_container_width=True):
        try:
            proposal_json = json.loads(raw_json)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")


if input_method == "Fill verified proposal form":
    st.markdown("## Verified Proposal Form")

    with st.form("proposal_form"):
        st.markdown("### 1. Customer Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            proposalno = st.text_input("Proposal No", value="")
            full_name = st.text_input("Full Name", value="Example Customer")
            name_with_initials = st.text_input("Name with Initials", value="E. Customer")
        with c2:
            mobile = st.text_input("Mobile", value="0771234567")
            email = st.text_input("Email", value="customer@example.com")
            preferred_language = st.selectbox("Preferred Language", ["English", "Sinhala", "Tamil"], index=0)
        with c3:
            age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
            date_of_birth = st.text_input("Date of Birth", value="1991-01-01")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"], index=1)

        st.markdown("### 2. Proposal / Financial Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            address = st.text_area("Address", value="Verified customer address", height=80)
            occupation = st.text_input("Occupation", value="ACCOUNTANT")
            company_or_institution = st.text_input("Company / Institution", value="Example Company")
        with c2:
            monthly_income = st.selectbox(
                "Monthly Income",
                [
                    "0-50000",
                    "50001-100000",
                    "100001-150000",
                    "150001-200000",
                    "200001-300000",
                    "300001-500000",
                    "500001-1000000",
                ],
                index=4,
            )
            plan_type = st.text_input("Plan Type", value="Life Insurance Plan")
            sum_insured = st.number_input("Sum Insured", min_value=100000, max_value=100000000, value=1000000, step=100000)
        with c3:
            premium_amount = st.number_input("Premium Amount", min_value=0, value=0, step=1000)
            payment_frequency = st.selectbox("Payment Frequency", ["Monthly", "Quarterly", "Half-yearly", "Yearly"], index=0)
            payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Bank Transfer", "Standing Order"], index=2)
            riders = st.multiselect("Riders", ["Accidental Death Benefit", "Critical Illness", "Hospital Cash", "Waiver of Premium"])

        st.markdown("### 3. Previous Insurance / Habits / Risk Questions")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            has_existing_or_previous_policy = yn_select("Existing / Previous Policy")
            has_declined_or_postponed_policy = yn_select("Declined / Postponed Policy")
        with c2:
            smoker = yn_select("Smoker")
            alcohol = yn_select("Alcohol")
        with c3:
            hazardous_occupation = yn_select("Hazardous Occupation")
            hazardous_sport = yn_select("Hazardous Sport")
        with c4:
            criminal_offence = yn_select("Criminal Offence")
            threat_on_life = yn_select("Threat on Life")

        st.markdown("### 4. Physical / Medical / Family History")

        c1, c2, c3 = st.columns(3)
        with c1:
            height_cm = st.number_input("Height cm", min_value=0, value=170, step=1)
            weight_kg = st.number_input("Weight kg", min_value=0, value=70, step=1)
            has_family_medical_history = yn_select("Family Medical History")
            visited_doctor_last_3_years = yn_select("Visited Doctor Last 3 Years")
        with c2:
            had_medical_condition_or_injury = yn_select("Medical Condition / Injury")
            had_operation_xray_or_hospital_test = yn_select("Operation / X-ray / Hospital Test")
            overnight_hospital_stay = yn_select("Overnight Hospital Stay")
            currently_taking_treatment_or_drugs = yn_select("Currently Taking Treatment / Drugs")
        with c3:
            mental_or_neurological_condition = yn_select("Mental / Neurological Condition")
            physical_disability_or_defect = yn_select("Physical Disability / Defect")
            absent_from_work_due_to_health = yn_select("Absent from Work Due to Health")
            medical_details_text = st.text_area("Medical Details Text", value="", height=90)

        st.markdown("### 5. Specific Disease Questions")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            heart_or_blood_pressure_or_stroke = yn_select("Heart / Blood Pressure / Stroke")
            kidney_or_urinary_or_eyes_ears = yn_select("Kidney / Urinary / Eyes / Ears")
        with c2:
            brain_or_nervous_or_mental = yn_select("Brain / Nervous / Mental")
            cancer_or_tumour = yn_select("Cancer / Tumour")
        with c3:
            multiple_sclerosis_or_arthritis_or_rheumatism = yn_select("MS / Arthritis / Rheumatism")
            hepatitis_aids_or_related_condition = yn_select("Hepatitis / AIDS Related")
        with c4:
            respiratory_or_asthma_or_bronchitis = yn_select("Respiratory / Asthma / Bronchitis")
            digestive_gall_bladder_liver_ulcer_bleeding = yn_select("Digestive / Liver / Ulcer / Bleeding")

        st.markdown("### 6. Female and Branch Checks")

        c1, c2, c3 = st.columns(3)
        with c1:
            pregnant_present = yn_select("Pregnant Present")
            pregnancy_complication_history = yn_select("Pregnancy Complication History")
            female_organ_disorder = yn_select("Female Organ Disorder")
        with c2:
            q11_family_question_yes = yn_select("Branch Q11 Family Question Yes")
            q12_medical_question_yes = yn_select("Branch Q12 Medical Question Yes")
        with c3:
            q13_additional_question_yes = yn_select("Branch Q13 Additional Question Yes")
            q14_female_question_yes = yn_select("Branch Q14 Female Question Yes")

        submit = st.form_submit_button("Run LLM Rule RAG Agent", use_container_width=True)

    if submit:
        form = {
            "proposalno": proposalno,
            "full_name": full_name,
            "name_with_initials": name_with_initials,
            "mobile": mobile,
            "email": email,
            "preferred_language": preferred_language,
            "age": age,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "marital_status": marital_status,
            "address": address,
            "occupation": occupation,
            "company_or_institution": company_or_institution,
            "monthly_income": monthly_income,
            "plan_type": plan_type,
            "sum_insured": sum_insured,
            "premium_amount": premium_amount,
            "payment_frequency": payment_frequency,
            "payment_method": payment_method,
            "riders": riders,
            "has_existing_or_previous_policy": has_existing_or_previous_policy,
            "has_declined_or_postponed_policy": has_declined_or_postponed_policy,
            "smoker": smoker,
            "alcohol": alcohol,
            "hazardous_occupation": hazardous_occupation,
            "hazardous_sport": hazardous_sport,
            "criminal_offence": criminal_offence,
            "threat_on_life": threat_on_life,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "has_family_medical_history": has_family_medical_history,
            "visited_doctor_last_3_years": visited_doctor_last_3_years,
            "had_medical_condition_or_injury": had_medical_condition_or_injury,
            "had_operation_xray_or_hospital_test": had_operation_xray_or_hospital_test,
            "overnight_hospital_stay": overnight_hospital_stay,
            "currently_taking_treatment_or_drugs": currently_taking_treatment_or_drugs,
            "mental_or_neurological_condition": mental_or_neurological_condition,
            "physical_disability_or_defect": physical_disability_or_defect,
            "absent_from_work_due_to_health": absent_from_work_due_to_health,
            "medical_details_text": medical_details_text,
            "heart_or_blood_pressure_or_stroke": heart_or_blood_pressure_or_stroke,
            "kidney_or_urinary_or_eyes_ears": kidney_or_urinary_or_eyes_ears,
            "brain_or_nervous_or_mental": brain_or_nervous_or_mental,
            "cancer_or_tumour": cancer_or_tumour,
            "multiple_sclerosis_or_arthritis_or_rheumatism": multiple_sclerosis_or_arthritis_or_rheumatism,
            "hepatitis_aids_or_related_condition": hepatitis_aids_or_related_condition,
            "respiratory_or_asthma_or_bronchitis": respiratory_or_asthma_or_bronchitis,
            "digestive_gall_bladder_liver_ulcer_bleeding": digestive_gall_bladder_liver_ulcer_bleeding,
            "pregnant_present": pregnant_present,
            "pregnancy_complication_history": pregnancy_complication_history,
            "female_organ_disorder": female_organ_disorder,
            "q11_family_question_yes": q11_family_question_yes,
            "q12_medical_question_yes": q12_medical_question_yes,
            "q13_additional_question_yes": q13_additional_question_yes,
            "q14_female_question_yes": q14_female_question_yes,
        }

        proposal_json = build_proposal_json(form)


# =========================================================
# RUN AGENT
# =========================================================
if proposal_json is not None:
    st.info("Verified proposal JSON submitted to the LLM Rule RAG STP/NSTP agent.")

    with st.spinner("Running agent..."):
        output = run_stp_nstp_agent(
            proposal_json=proposal_json,
            save_output=save_output,
            use_llm=use_llm,
        )

    st.session_state["last_proposal_json"] = proposal_json
    st.session_state["last_output"] = output


if "last_output" in st.session_state:
    render_output(st.session_state["last_output"])
else:
    st.info("Fill the verified proposal form or paste verified proposal JSON, then run the agent.")