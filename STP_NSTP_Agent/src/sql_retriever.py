# =========================================================
# src/sql_retriever.py
# Custom SQL tool for retrieving similar past proposal data
#
# Important:
# - This module does NOT retrieve rules.
# - All tables with names containing "rule" are excluded.
# - It retrieves proposal details, need analysis, medical details,
#   questionnaire answers, and underwriter remarks from historical data.
# =========================================================

from typing import Any, Dict, List, Optional
import json
import sqlite3
import pandas as pd

from langchain_core.tools import tool

from src.config import DB_PATH
from src.utils import (
    clean_text,
    clean_value,
    parse_age,
    parse_income_midpoint,
    parse_numeric,
    df_to_records,
    json_safe,
    build_retrieval_keywords,
)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    try:
        df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            conn,
            params=[table_name]
        )
        return len(df) > 0
    except Exception:
        return False


def safe_read_sql(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[List[Any]] = None
) -> pd.DataFrame:
    if params is None:
        params = []
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def get_all_table_names(conn: sqlite3.Connection) -> List[str]:
    df = safe_read_sql(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    if len(df) == 0:
        return []
    return [
        str(x)
        for x in df["name"].tolist()
        if "rule" not in str(x).lower()
    ]


def get_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    try:
        info = pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)
        return info["name"].tolist()
    except Exception:
        return []


def find_first_existing_table(conn: sqlite3.Connection, candidates: List[str]) -> Optional[str]:
    all_tables = get_all_table_names(conn)
    lower_map = {t.lower(): t for t in all_tables}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for table in all_tables:
        table_l = table.lower()
        for cand in candidates:
            if cand.lower() in table_l:
                return table

    return None


def find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in cols:
        col_l = col.lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col

    return None


def get_proposal_identifiers(proposal_json: Dict[str, Any]) -> Dict[str, Optional[str]]:
    metadata = proposal_json.get("proposal_metadata", {})
    personal = proposal_json.get("personal_details", {})
    contact = proposal_json.get("customer_contact", {})

    return {
        "proposalno": clean_value(metadata.get("proposalno")),
        "quoteno": clean_value(metadata.get("quoteno")),
        "memberid": clean_value(metadata.get("memberid")),
        "nic": clean_value(personal.get("nic")),
        "mobile": clean_value(contact.get("mobile")),
        "email": clean_value(contact.get("email")),
    }


def retrieve_exact_or_linked_records(
    conn: sqlite3.Connection,
    proposal_json: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Try to find exact/linked records by proposalno, quoteno, memberid, NIC, mobile, or email.
    For new proposals this usually returns no exact match.
    """
    ids = get_proposal_identifiers(proposal_json)
    result = {
        "matched": False,
        "searched_identifiers": ids,
        "records": {}
    }

    tables = get_all_table_names(conn)

    for table in tables:
        cols = get_columns(conn, table)
        if not cols:
            continue

        conditions = []
        params = []

        possible_matches = [
            ("proposalno", ["proposalno", "proposal_no", "policy_proposalno"]),
            ("quoteno", ["quoteno", "quote_no", "quotation_no"]),
            ("memberid", ["memberid", "member_id", "contactid", "contact_id"]),
            ("nic", ["nic", "nic_no", "identity_no", "national_identity_card_number"]),
            ("mobile", ["mobile", "mobile_no", "phone"]),
            ("email", ["email", "email_address"]),
        ]

        for id_key, col_candidates in possible_matches:
            value = ids.get(id_key)
            if not value:
                continue

            col = find_col(cols, col_candidates)
            if col:
                conditions.append(f'LOWER(CAST("{col}" AS TEXT)) = ?')
                params.append(value.lower())

        if not conditions:
            continue

        query = f"""
        SELECT *, '{table}' AS source_table
        FROM "{table}"
        WHERE {" OR ".join(conditions)}
        LIMIT 20
        """

        df = safe_read_sql(conn, query, params)
        if len(df) > 0:
            result["matched"] = True
            result["records"][table] = df_to_records(df, max_rows=20)

    return json_safe(result)


def build_similarity_where_clause(cols: List[str], proposal_json: Dict[str, Any]) -> tuple[str, List[Any]]:
    personal = proposal_json.get("personal_details", {})
    proposal = proposal_json.get("proposal_details", {})

    age = parse_age(personal.get("age"))
    gender = clean_value(personal.get("gender"))
    occupation = clean_value(personal.get("occupation"))
    income_mid = parse_income_midpoint(personal.get("monthly_income"))
    sum_insured = parse_numeric(proposal.get("sum_insured"))

    conditions = []
    params: List[Any] = []

    age_col = find_col(cols, ["age"])
    if age is not None and age_col:
        conditions.append(f'CAST("{age_col}" AS INTEGER) BETWEEN ? AND ?')
        params.extend([max(age - 5, 0), age + 5])

    gender_col = find_col(cols, ["gender", "sex"])
    if gender and gender_col:
        conditions.append(f'LOWER(CAST("{gender_col}" AS TEXT)) = ?')
        params.append(gender.lower())

    occ_col = find_col(cols, ["occupation", "occupationcode", "occupation_code", "job"])
    if occupation and occ_col:
        occ_first = occupation.split()[0].lower()
        conditions.append(f'(LOWER(CAST("{occ_col}" AS TEXT)) LIKE ? OR LOWER(CAST("{occ_col}" AS TEXT)) LIKE ?)')
        params.extend([f"%{occupation.lower()}%", f"%{occ_first}%"])

    income_col = find_col(cols, ["monthly_income", "income", "salary"])
    if income_mid is not None and income_col:
        low = income_mid * 0.50
        high = income_mid * 1.50
        conditions.append(f'CAST("{income_col}" AS REAL) BETWEEN ? AND ?')
        params.extend([low, high])

    sum_col = find_col(cols, ["sum_insured", "sumassured", "sum_assured", "basic_sum_assured"])
    if sum_insured is not None and sum_col:
        low = sum_insured * 0.50
        high = sum_insured * 1.50
        conditions.append(f'CAST("{sum_col}" AS REAL) BETWEEN ? AND ?')
        params.extend([low, high])

    if not conditions:
        return "", []

    return " OR ".join(conditions), params


def retrieve_similar_base_cases(
    conn: sqlite3.Connection,
    proposal_json: Dict[str, Any],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve similar cases mainly from needanalysis/proposal tables.
    The query is dynamic because column names can vary.
    """
    candidate_tables = [
        "needanalysis",
        "proposaldetails_1",
        "proposaldetails",
        "proposal_details"
    ]

    base_rows: List[Dict[str, Any]] = []

    for table_candidate in candidate_tables:
        table = find_first_existing_table(conn, [table_candidate])
        if not table:
            continue

        cols = get_columns(conn, table)
        where_clause, params = build_similarity_where_clause(cols, proposal_json)

        if where_clause:
            query = f"""
            SELECT *, '{table}' AS source_table
            FROM "{table}"
            WHERE {where_clause}
            LIMIT ?
            """
            params.append(limit)
        else:
            query = f"""
            SELECT *, '{table}' AS source_table
            FROM "{table}"
            LIMIT ?
            """
            params = [limit]

        df = safe_read_sql(conn, query, params)
        base_rows.extend(df_to_records(df, max_rows=limit))

        if len(base_rows) >= limit:
            break

    return json_safe(base_rows[:limit])


def collect_linked_records_for_case(
    conn: sqlite3.Connection,
    base_case: Dict[str, Any],
    max_records_each: int = 8
) -> Dict[str, Any]:
    """
    Enrich each similar case with linked proposal, medical, questionnaire,
    and underwriter remark records.
    """
    all_tables = get_all_table_names(conn)

    def value_from_keys(keys: List[str]) -> Optional[str]:
        lower_base = {str(k).lower(): v for k, v in base_case.items()}
        for key in keys:
            if key.lower() in lower_base:
                value = clean_value(lower_base[key.lower()])
                if value:
                    return value
        return None

    proposalno = value_from_keys(["proposalno", "proposal_no", "policy_proposalno"])
    quoteno = value_from_keys(["quoteno", "quote_no", "quotation_no"])
    memberid = value_from_keys(["memberid", "member_id", "contactid", "contact_id"])

    linked = {
        "identifiers": {
            "proposalno": proposalno,
            "quoteno": quoteno,
            "memberid_or_contactid": memberid
        },
        "proposal_details": [],
        "medical_details": [],
        "questionnaire_answers": [],
        "underwriter_remarks": []
    }

    for table in all_tables:
        cols = get_columns(conn, table)
        table_l = table.lower()

        identifier_conditions = []
        params = []

        if proposalno:
            col = find_col(cols, ["proposalno", "proposal_no", "policy_proposalno"])
            if col:
                identifier_conditions.append(f'LOWER(CAST("{col}" AS TEXT)) = ?')
                params.append(proposalno.lower())

        if quoteno:
            col = find_col(cols, ["quoteno", "quote_no", "quotation_no"])
            if col:
                identifier_conditions.append(f'LOWER(CAST("{col}" AS TEXT)) = ?')
                params.append(quoteno.lower())

        if memberid:
            col = find_col(cols, ["memberid", "member_id", "contactid", "contact_id"])
            if col:
                identifier_conditions.append(f'LOWER(CAST("{col}" AS TEXT)) = ?')
                params.append(memberid.lower())

        if not identifier_conditions:
            continue

        query = f"""
        SELECT *, '{table}' AS source_table
        FROM "{table}"
        WHERE {" OR ".join(identifier_conditions)}
        LIMIT ?
        """
        params.append(max_records_each)

        df = safe_read_sql(conn, query, params)
        records = df_to_records(df, max_rows=max_records_each)
        if not records:
            continue

        if "medical" in table_l:
            linked["medical_details"].extend(records)
        elif "question" in table_l or "quest" in table_l:
            linked["questionnaire_answers"].extend(records)
        elif "uw" in table_l or "remark" in table_l or "underwriter" in table_l:
            linked["underwriter_remarks"].extend(records)
        elif "proposal" in table_l or "need" in table_l:
            linked["proposal_details"].extend(records)

    return json_safe(linked)


def search_underwriter_remarks_by_keywords(
    conn: sqlite3.Connection,
    keywords: List[str],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search historical free-text UW/remarks tables for similar document requests.
    """
    if not keywords:
        return []

    all_tables = get_all_table_names(conn)
    remark_tables = [
        t for t in all_tables
        if any(x in t.lower() for x in ["uw", "remark", "underwriter"])
    ]

    results: List[Dict[str, Any]] = []

    for table in remark_tables:
        cols = get_columns(conn, table)
        text_cols = [
            c for c in cols
            if any(x in c.lower() for x in ["remark", "comment", "note", "description", "reason", "requirement"])
        ]

        if not text_cols:
            text_cols = cols[:8]

        like_blocks = []
        params = []

        for keyword in keywords[:12]:
            keyword = clean_text(keyword)
            if not keyword:
                continue

            col_checks = []
            for col in text_cols:
                col_checks.append(f'LOWER(CAST("{col}" AS TEXT)) LIKE ?')
                params.append(f"%{keyword.lower()}%")
            if col_checks:
                like_blocks.append("(" + " OR ".join(col_checks) + ")")

        if not like_blocks:
            continue

        query = f"""
        SELECT *, '{table}' AS source_table
        FROM "{table}"
        WHERE {" OR ".join(like_blocks)}
        LIMIT ?
        """
        params.append(limit)

        df = safe_read_sql(conn, query, params)
        results.extend(df_to_records(df, max_rows=limit))

    return json_safe(results[:limit])


def retrieve_sql_past_case_context(
    proposal_json: Dict[str, Any],
    rule_check: Dict[str, Any],
    limit: int = 8
) -> Dict[str, Any]:
    """
    Main SQL retrieval function.

    Output contains only historical proposal context.
    Rules are intentionally excluded.
    """
    if not DB_PATH.exists():
        return {
            "tool_name": "sql_past_case_retriever",
            "status": "error",
            "error": f"Database not found: {DB_PATH}",
            "tables_used": [],
            "exact_or_linked_records": {},
            "similar_cases": [],
            "keyword_remark_matches": [],
            "retrieval_note": "SQL retrieval failed because database file was not found."
        }

    conn = sqlite3.connect(DB_PATH)

    try:
        tables_available = get_all_table_names(conn)

        exact_or_linked = retrieve_exact_or_linked_records(conn, proposal_json)
        similar_base_cases = retrieve_similar_base_cases(conn, proposal_json, limit=limit)

        enriched_cases = []
        for base_case in similar_base_cases:
            linked = collect_linked_records_for_case(conn, base_case, max_records_each=6)
            enriched_cases.append({
                "base_case": base_case,
                "linked_records": linked
            })

        keywords = build_retrieval_keywords(proposal_json, rule_check)
        keyword_remark_matches = search_underwriter_remarks_by_keywords(
            conn,
            keywords=keywords,
            limit=limit
        )

        return json_safe({
            "tool_name": "sql_past_case_retriever",
            "status": "success",
            "tables_available_without_rules": tables_available,
            "tables_used_note": (
                "All tables with names containing 'rule' are excluded. "
                "The tool searches proposal, need analysis, medical, questionnaire, "
                "and underwriter remark tables when available."
            ),
            "exact_or_linked_records": exact_or_linked,
            "similar_cases": enriched_cases,
            "keyword_remark_matches": keyword_remark_matches,
            "retrieval_keywords": keywords,
            "retrieval_note": (
                "SQL context is historical supporting evidence only. "
                "It is used for deciding needed documents and loading handoff, "
                "not for retrieving company STP/NSTP rules."
            )
        })
    finally:
        conn.close()


@tool
def sql_past_case_retriever(tool_input_json: str) -> str:
    """
    Custom SQL tool for the LLM-first STP/NSTP agent.

    Input JSON format:
    {
      "proposal_json": {...},
      "rule_check": {...},
      "limit": 8
    }

    Output:
    Historical similar cases, medical records, questionnaire answers,
    and underwriter remarks. Rules are excluded.
    """
    try:
        data = json.loads(tool_input_json)
        proposal_json = data.get("proposal_json", {})
        rule_check = data.get("rule_check", {})
        limit = int(data.get("limit", 8))
        result = retrieve_sql_past_case_context(
            proposal_json=proposal_json,
            rule_check=rule_check,
            limit=limit
        )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "tool_name": "sql_past_case_retriever",
            "status": "error",
            "error": str(e)
        }, indent=2, ensure_ascii=False)
