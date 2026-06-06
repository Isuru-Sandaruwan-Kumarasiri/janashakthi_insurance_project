import json
import os
import re
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter

# ── Path resolution ───────────────────────────────────────────────────────────
PROJECT_DIR    = current = Path(__file__).resolve().parent.parent
DB_PATH        = PROJECT_DIR / "database" / "underwritingsystem.db"
VECTOR_CONFIG  = PROJECT_DIR / "data" / "processed" / "latestvectorstoreconfig.json"
OUTPUT_DIR     = PROJECT_DIR / "data" / "processed" / "agentoutputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOADING_OUTPUT = OUTPUT_DIR / "finalloadingdecision.json"

load_dotenv(dotenv_path=PROJECT_DIR / ".env", override=True)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTERMODEL", "openrouter/owl-alpha")
LOADING_AGENT_VER   = "2.0.0"
MAX_SINGLE_LOADING  = 200
TOTAL_LOADING_CAP   = 100
COMBINATION_METHOD  = "ADDITIVE"
VECTOR_K            = 8
VALID_LOADING_TYPES = {"HEALTH_LOADING", "LIFESTYLE_LOADING", "OCCUPATION_LOADING"}
QUESTIONNAIRE_TABLES = ["Questionaire20201231", "Questionaire20231231", "Questionaire20251231"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean(value) -> str:
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except Exception: pass
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else re.sub(r"\s+", " ", text)

def _yes(value) -> bool:
    return _clean(value).lower() in {"yes", "true", "1", "y", "checked"}

def _parse_income_mid(value) -> float:
    text = _clean(value).replace(",", "")
    if not text: return 0.0
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", text)]
    return sum(nums) / len(nums) if nums else 0.0

def _get_numeric(value) -> float:
    text = _clean(value).replace(",", "")
    nums = re.findall(r"\d+\.?\d*", text)
    return float(nums[0]) if nums else 0.0

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text).strip()
    return text

def _safe_parse_json(raw) -> tuple:
    if not raw: return None, "LLM returned empty response."
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), None
            except Exception:
                pass
        return None, f"JSON parse error: {e}"

def _table_exists(conn, name: str) -> bool:
    try:
        r = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name=?", conn, params=(name,))
        return len(r) > 0
    except Exception: return False

def _safe_sql(conn, query, params=None):
    try: return pd.read_sql_query(query, conn, params=params or [])
    except Exception: return pd.DataFrame()

def _get_cols(conn, table: str) -> list:
    try:
        return pd.read_sql_query(f"PRAGMA table_info('{table}')", conn)["name"].tolist()
    except Exception: return []

# ── Questionnaire retrieval ───────────────────────────────────────────────────
def _get_questionnaire_answers(proposalno: str, quoteno=None, memberid=None) -> list:
    results = []
    if not DB_PATH.exists(): return results
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row.lower() for row in cursor.fetchall()}
    for tbl in QUESTIONNAIRE_TABLES:
        if tbl.lower() not in existing: continue
        cols = _get_cols(conn, tbl)
        conditions, params = [], []
        for col, val in [("proposalno", proposalno), ("quoteno", quoteno), ("memberid", memberid)]:
            if val and col in cols:
                conditions.append(f"{col} = ?"); params.append(val)
        if not conditions: continue
        try:
            cursor.execute(f"SELECT * FROM {tbl} WHERE {' OR '.join(conditions)}", params)
            for row in cursor.fetchall():
                d = dict(row); d["_source_table"] = tbl; results.append(d)
        except sqlite3.Error: pass
    conn.close()
    return results

# ── Loading rules retrieval ───────────────────────────────────────────────────
def _retrieve_loading_rules(loading_types: list, occupation: str, has_medical: bool) -> list:
    if not DB_PATH.exists(): return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    if not _table_exists(conn, "rulesmasterclean"):
        conn.close(); return []
    conditions, params = [], []
    for lt in loading_types:
        conditions.append("UPPER(possibleloadingtypes) LIKE ?")
        params.append(f"%{lt.upper()}%")
    for term in occupation.split()[:4]:
        if len(term) > 3:
            conditions.append("(LOWER(ruledescription) LIKE ? OR LOWER(conditiontext) LIKE ?)")
            params.extend([f"%{term.lower()}%", f"%{term.lower()}%"])
    if has_medical:
        conditions.append("(LOWER(ruledescription) LIKE '%medical%' OR LOWER(conditiontext) LIKE '%medical%' OR LOWER(ruletype) LIKE '%medical%')")
    where = "(" + " OR ".join(conditions) + ")" if conditions else "possibleloadingtypes IS NOT NULL AND TRIM(possibleloadingtypes) != ''"
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT ruleid, ruletype, ruledescription, conditiontext, suggesteddecision, possibleloadingtypes, ruleexplanation, comments, fieldstovalidate FROM rulesmasterclean WHERE {where} LIMIT 30", params)
        rows = [dict(r) for r in cursor.fetchall()]
    except sqlite3.Error: rows = []
    conn.close()
    return rows

# ── Vector retrieval ───────────────────────────────────────────────────────────
def _resolve_chroma_path(chromadir: str) -> Path:
    p = Path(chromadir)
    if p.exists(): return p
    alt = PROJECT_DIR / "vectorstore" / p.name
    if alt.exists(): return alt
    return PROJECT_DIR / chromadir if (PROJECT_DIR / chromadir).exists() else p

def _load_vector_store():
    if not VECTOR_CONFIG.exists(): return None
    with open(VECTOR_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    chroma_dir      = _resolve_chroma_path(cfg["chromadir"])
    collection_name = cfg.get("collectionname", "underwritingknowledge")
    embedding_model = cfg.get("embeddingmodel", "nomic-embed-text")
    if not chroma_dir.exists(): return None
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=str(chroma_dir))

def _retrieve_vector_docs(loading_types: list, occupation: str, has_medical: bool,
                          has_lifestyle: bool, has_hobby: bool, smoker: bool, alcohol: bool) -> list:
    vectorstore = _load_vector_store()
    if vectorstore is None: return []
    parts = ["premium loading underwriting"]
    if loading_types: parts.append(" ".join(lt.replace("_", " ").lower() for lt in loading_types))
    if occupation:    parts.append(f"occupation {occupation}")
    if has_medical:   parts.append("medical history disease illness loading")
    if smoker:        parts.append("smoker tobacco loading")
    if alcohol:       parts.append("alcohol drinker loading")
    if has_hobby:     parts.append("hazardous hobby extreme sport loading")
    query = " ".join(parts)
    docs  = vectorstore.similarity_search(query, k=VECTOR_K)
    return [{"pagecontent": d.page_content, "metadata": d.metadata} for d in docs]

# ── Prompt builders ───────────────────────────────────────────────────────────
OUTPUT_SCHEMA = """
Return a JSON object with these top-level keys:
{
  "loadingassignments": [
    {
      "loadingtype": "HEALTH_LOADING | LIFESTYLE_LOADING | OCCUPATION_LOADING",
      "loadingpercentage": <positive integer 0-200 or null>,
      "flatextraper1000": <positive number or null>,
      "reason": "<why this loading is recommended>",
      "matchedruleid": "<actual ruleid from provided rules, or null>",
      "matchedruledescription": "<matched rule description, or null>",
      "supportingevidence": {
        "questionnaireitems": [{"qid": "", "qtext": "", "answer": ""}],
        "vectordocs": [{"sourcetype": "", "sourcename": "", "snippet": ""}],
        "similarcases": [{"casereference": "", "whyrelevant": ""}]
      },
      "confidence": "HIGH | MEDIUM | LOW"
    }
  ],
  "declinerecommendation": {
    "shoulddecline": <true|false>,
    "reason": "<string or empty>"
  },
  "importantnote": "<uncertainty, evidence limitations, or need for human review>"
}
Rules:
- Return ONLY valid JSON. No markdown. No preamble.
- ONE entry per distinct loading type. Do not duplicate loading types.
- loadingpercentage must be a positive integer between 0 and 200, or null.
- matchedruleid must reference an actual ruleid from the rules provided, or null.
- Do NOT invent rules not in the provided evidence.
""".strip()

def _build_loading_prompt(proposal_summary, possible_loading_types, quest_block,
                          rules_block, vector_block, upstream, risk_factors,
                          violated_rules_upstream, required_medical) -> str:
    return f"""You are a senior life insurance underwriting specialist at Janashakthi Insurance.
Your task is to recommend PREMIUM LOADINGS for a proposal flagged for loading review.

### UPSTREAM DECISION SUMMARY
STP/NSTP Decision : {upstream.get('decision', 'N/A')}
Confidence        : {upstream.get('confidence', 'N/A')}
Requires Loading  : {upstream.get('reviewflags', {}).get('loadingreviewrequired', True)}
Possible Types    : {', '.join(possible_loading_types)}
Main Reasons      : {'; '.join(upstream.get('mainreasons', []))}

### PROPOSAL SUMMARY
{json.dumps(proposal_summary, ensure_ascii=False, indent=2)}

### RISK FACTORS (pre-computed by upstream agent)
{json.dumps(risk_factors, ensure_ascii=False, indent=2)}

### UPSTREAM VIOLATED RULES
{json.dumps(violated_rules_upstream or [], ensure_ascii=False, indent=2)}

### REQUIRED MEDICAL REPORTS
{chr(10).join(f'  - {r}' for r in required_medical) if required_medical else 'None.'}

### QUESTIONNAIRE EVIDENCE
{quest_block}

### LOADING RULES FROM DATABASE
{rules_block}

### VECTOR RAG EVIDENCE
{vector_block}

### OUTPUT SCHEMA AND INSTRUCTIONS
{OUTPUT_SCHEMA}

Based on all evidence above, recommend appropriate premium loadings.
Each loadingpercentage is a percentage surcharge on the base premium (e.g. 25 means +25%).
Justify every recommendation with specific risk factors and/or rules provided.
If evidence is insufficient, set loadingpercentage to null, confidence to LOW, and explain in importantnote.
"""

def _format_questionnaire(rows: list, decl: dict) -> str:
    lines = []
    if decl:
        lines.append("Questionnaire Declarations from OCR input:")
        for k, v in decl.items(): lines.append(f"  {k}: {v}")
    if rows:
        lines.append("Questionnaire Rows from Database:")
        for row in rows[:20]:
            tbl = row.get("_source_table", "unknown")
            filtered = {k: v for k, v in row.items() if k != "_source_table" and v not in (None, "", "NA")}
            lines.append(f"  [{tbl}] {json.dumps(filtered, ensure_ascii=False)[:300]}")
    return "\n".join(lines) if lines else "No questionnaire evidence available."

def _format_rules(rules: list) -> str:
    if not rules: return "No loading rules retrieved from database."
    lines = ["Relevant Rules from rulesmasterclean:"]
    for r in rules:
        lines.extend([
            f"  Rule: {r.get('ruleid','?')} [{r.get('ruletype','?')}]",
            f"  {str(r.get('ruledescription',''))[:120]}",
            f"  Condition: {str(r.get('conditiontext',''))[:120]}",
            f"  Decision: {r.get('suggesteddecision','')}",
            f"  LoadingType: {r.get('possibleloadingtypes','')}",
            f"  Explanation: {str(r.get('ruleexplanation',''))[:120]}",
        ])
    return "\n".join(lines)

def _format_vector_docs(docs: list) -> str:
    if not docs: return "No vector documents retrieved."
    lines = ["Vector RAG Evidence:"]
    for i, doc in enumerate(docs[:10], 1):
        meta    = doc.get("metadata", {})
        stype   = meta.get("sourcetype", meta.get("source_type", "unknown"))
        sname   = meta.get("sourcename", meta.get("source_name", meta.get("ruleid", "unknown")))
        snippet = doc.get("pagecontent", "")[:300].replace("\n", " ")
        lines.append(f"  [{i}] sourcetype={stype}, sourcename={sname}: {snippet}")
    return "\n".join(lines)

# ── Validate assignments ───────────────────────────────────────────────────────
def _validate_assignments(assignments: list) -> list:
    validated, seen = [], set()
    for item in assignments:
        if not isinstance(item, dict): continue
        lt = str(item.get("loadingtype", "")).upper().strip()
        if lt not in VALID_LOADING_TYPES: continue
        if lt in seen: continue
        seen.add(lt)
        pct = item.get("loadingpercentage")
        if pct is not None:
            try:
                pct = float(pct)
                if pct < 0: pct = None
                elif pct > MAX_SINGLE_LOADING: pct = float(MAX_SINGLE_LOADING)
            except (TypeError, ValueError): pct = None
        flat = item.get("flatextraper1000")
        if flat is not None:
            try:
                flat = float(flat)
                if flat < 0: flat = None
            except (TypeError, ValueError): flat = None
        validated.append({
            "loadingtype"           : lt,
            "loadingpercentage"     : pct,
            "flatextraper1000"      : flat,
            "reason"                : str(item.get("reason", "")),
            "matchedruleid"         : item.get("matchedruleid"),
            "matchedruledescription": item.get("matchedruledescription"),
            "supportingevidence"    : item.get("supportingevidence", {"questionnaireitems": [], "vectordocs": [], "similarcases": []}),
            "confidence"            : str(item.get("confidence", "LOW")).upper(),
        })
    return validated

def _combine_loadings(assignments: list) -> tuple:
    numeric_pcts = [a["loadingpercentage"] for a in assignments if a["loadingpercentage"] is not None]
    total = sum(numeric_pcts)
    cap_applied, cap_details = False, None
    if total > TOTAL_LOADING_CAP:
        cap_applied = True
        cap_details = f"Total loading capped at {TOTAL_LOADING_CAP}% per company guideline (computed {total:.0f}%)."
        total = float(TOTAL_LOADING_CAP)
    return round(total, 2), cap_applied, cap_details

def _evaluate_decline(decline_from_llm: dict, rules: list, total_pct: float) -> dict:
    if decline_from_llm.get("shoulddecline"):
        return {"shoulddecline": True, "reason": decline_from_llm.get("reason", "LLM recommended decline.")}
    decline_rules = [r for r in rules if str(r.get("suggesteddecision", "")).upper() == "DECLINE"]
    if decline_rules:
        dr = decline_rules
        return {"shoulddecline": True, "reason": f"Rule {dr.get('ruleid','?')} {str(dr.get('ruledescription',''))[:80]} recommends DECLINE."}
    if total_pct >= MAX_SINGLE_LOADING:
        return {"shoulddecline": True, "reason": f"Combined loading {total_pct}% exceeds maximum insurable threshold. Recommended for decline pending human review."}
    return {"shoulddecline": False, "reason": ""}

# ── Main entry point ───────────────────────────────────────────────────────────
def run_loading_agent(stp_output: dict) -> dict:
    """
    Main entry point. Accepts the STP/NSTP agent output dict, returns the loading decision dict.
    """
    review_flags        = stp_output.get("reviewflags", {})
    loading_agent_input = stp_output.get("loadingagentinput", {})
    risk_factors        = loading_agent_input.get("riskfactors", {})
    proposal_summary    = loading_agent_input.get("proposalsummary", {})
    possible_loading_types = risk_factors.get("possibleloadingtypes", [])
    violated_rules_upstream = loading_agent_input.get("violatedrules", [])
    required_medical        = loading_agent_input.get("requiredmedicalreports", [])

    requires_loading = review_flags.get("loadingreviewrequired", False)

    if not requires_loading:
        empty = {
            "proposalreference"    : {"proposalno": proposal_summary.get("proposalno"), "quoteno": proposal_summary.get("quoteno"), "memberid": proposal_summary.get("memberid")},
            "loadingrequired"      : False,
            "loadingassignments"   : [],
            "totalloadingpercentage": 0,
            "combinationmethod"    : COMBINATION_METHOD,
            "capapplied"           : False,
            "capdetails"           : None,
            "declinerecommendation": {"shoulddecline": False, "reason": ""},
            "importantnote"        : "Upstream agent did not flag loading review as required.",
            "audittrail"           : {"timestamp": datetime.now(timezone.utc).isoformat(), "modelused": OPENROUTER_MODEL, "prompthash": None, "version": LOADING_AGENT_VER},
        }
        with open(LOADING_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=4, ensure_ascii=False)
        return empty

    # ── Extract fields ──────────────────────────────────────────────────────────
    proposalno      = _clean(proposal_summary.get("proposalno", ""))
    quoteno         = proposal_summary.get("quoteno")
    memberid        = proposal_summary.get("memberid")
    occupation      = _clean(proposal_summary.get("occupation", "")).lower()
    monthly_income  = _parse_income_mid(proposal_summary.get("monthlyincome", ""))
    sum_insured     = _get_numeric(str(proposal_summary.get("suminsured", "")))
    has_medical     = bool(risk_factors.get("medicalreviewrequired", False))
    has_lifestyle   = bool(risk_factors.get("smokeryes", False)) or bool(risk_factors.get("alcoholyes", False))
    has_hobby       = bool(risk_factors.get("hazardoussportyes", False))
    smoker_yes      = bool(risk_factors.get("smokeryes", False))
    alcohol_yes     = bool(risk_factors.get("alcoholyes", False))

    quest_decl = {
        "smoker"              : "Yes" if smoker_yes else "No",
        "alcohol"             : "Yes" if alcohol_yes else "No",
        "hazardoussport"      : "Yes" if has_hobby else "No",
        "hazardousoccupation" : "Yes" if risk_factors.get("hazardousoccupationyes") else "No",
        "highriskoccupation"  : "Yes" if risk_factors.get("highriskoccupationtext") else "No",
        "medicalreviewrequired": "Yes" if has_medical else "No",
        "requireddocuments"   : ", ".join(loading_agent_input.get("requireddocuments", [])) or None,
        "requiredmedicalreports": ", ".join(required_medical) or None,
    }

    # ── Retrieve context ────────────────────────────────────────────────────────
    questionnaire_rows = _get_questionnaire_answers(proposalno, quoteno, memberid)
    loading_rules      = _retrieve_loading_rules(possible_loading_types, occupation, has_medical)
    vector_docs        = _retrieve_vector_docs(possible_loading_types, occupation, has_medical,
                                               has_lifestyle, has_hobby, smoker_yes, alcohol_yes)

    # ── Format context blocks ───────────────────────────────────────────────────
    quest_block  = _format_questionnaire(questionnaire_rows, quest_decl)
    rules_block  = _format_rules(loading_rules)
    vector_block = _format_vector_docs(vector_docs)

    # ── Build prompt & call LLM ─────────────────────────────────────────────────
    final_prompt = _build_loading_prompt(
        proposal_summary, possible_loading_types, quest_block, rules_block, vector_block,
        stp_output, risk_factors, violated_rules_upstream, required_medical
    )
    prompt_hash = hashlib.sha256(final_prompt.encode()).hexdigest()[:16]

    try:
        llm = ChatOpenRouter(model=OPENROUTER_MODEL, api_key=OPENROUTER_API_KEY, temperature=0)
        response = llm.invoke(final_prompt)
        raw_response = response.content
    except Exception as e:
        raw_response = None
        parse_error  = str(e)
        parsed_llm   = None
    else:
        parsed_llm, parse_error = _safe_parse_json(raw_response)

    if parsed_llm is None:
        assignments_raw = []
        decline_raw     = {"shoulddecline": False, "reason": ""}
        important_note  = f"LLM call or JSON parse failed: {parse_error}. Manual underwriter review required."
    else:
        assignments_raw = parsed_llm.get("loadingassignments", [])
        decline_raw     = parsed_llm.get("declinerecommendation", {"shoulddecline": False, "reason": ""})
        important_note  = parsed_llm.get("importantnote", "")

    # ── Post-process ────────────────────────────────────────────────────────────
    assignments_validated       = _validate_assignments(assignments_raw)
    total_pct, cap_applied, cap_details = _combine_loadings(assignments_validated)
    decline_recommendation      = _evaluate_decline(decline_raw, loading_rules, total_pct)

    note_parts = [important_note]
    if cap_applied and cap_details:   note_parts.append(cap_details)
    if parse_error:                   note_parts.append(f"LLM JSON parse failed: {parse_error}")
    if not loading_rules:             note_parts.append("No matching loading rules found in database; recommendations rely on questionnaire evidence and vector RAG only.")

    final_output = {
        "proposalreference"     : {"proposalno": proposalno, "quoteno": quoteno, "memberid": memberid},
        "upstreamdecision"      : stp_output.get("decision"),
        "upstreamreviewflags"   : review_flags,
        "loadingrequired"       : True,
        "loadingassignments"    : assignments_validated,
        "totalloadingpercentage": total_pct,
        "combinationmethod"     : COMBINATION_METHOD,
        "capapplied"            : cap_applied,
        "capdetails"            : cap_details,
        "declinerecommendation" : decline_recommendation,
        "importantnote"         : " | ".join(p for p in note_parts if p),
        "audittrail"            : {
            "timestamp"  : datetime.now(timezone.utc).isoformat(),
            "modelused"  : OPENROUTER_MODEL,
            "prompthash" : prompt_hash,
            "promptchars": len(final_prompt),
            "parseerror" : parse_error,
            "version"    : LOADING_AGENT_VER,
        },
        "evidencesummary": {
            "questionnairerotscount": len(questionnaire_rows),
            "loadingrulescount"     : len(loading_rules),
            "vectordocscount"       : len(vector_docs),
        },
    }

    with open(LOADING_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    return final_output