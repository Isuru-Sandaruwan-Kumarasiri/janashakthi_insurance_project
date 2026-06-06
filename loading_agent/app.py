import streamlit as st
import json
import pandas as pd
from pathlib import Path
from src.loading_agent import run_loading_agent

# ── Path to existing STP/NSTP output ─────────────────────────────────────────
PROJECT_DIR      = Path(__file__).resolve().parent
STP_OUTPUT_PATH  = PROJECT_DIR / "data" / "processed" / "agentoutputs" / "binarystpnstpagentv4output.json"

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Janashakthi Loading Agent", page_icon="🏦", layout="wide")
st.title("🏦 Janashakthi — Premium Loading Assessment")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD STP/NSTP OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
st.header("📂 STP/NSTP Agent Output")

stp_result = None

tab_file, tab_paste = st.tabs(["📁 Load from File", "📥 Paste JSON"])

with tab_file:
    if STP_OUTPUT_PATH.exists():
        st.success(f"✅ Found: `{STP_OUTPUT_PATH}`")
        if st.button("Load from file", use_container_width=True):
            with open(STP_OUTPUT_PATH, "r", encoding="utf-8") as f:
                stp_result = json.load(f)
            st.session_state["stp_result"] = stp_result
    else:
        st.warning(f"⚠️ File not found at `{STP_OUTPUT_PATH}`. Use the Paste JSON tab instead.")

with tab_paste:
    st.info("Paste the full `binarystpnstpagentv4output.json` content here.")
    pasted = st.text_area("Paste STP/NSTP output JSON", height=300,
                          placeholder='{\n  "decision": "NSTP",\n  "reviewflags": {...},\n  ...\n}')
    if st.button("Load pasted JSON", use_container_width=True):
        try:
            stp_result = json.loads(pasted)
            st.session_state["stp_result"] = stp_result
            st.success("✅ JSON loaded successfully.")
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {e}")

# Restore from session if already loaded
if stp_result is None and "stp_result" in st.session_state:
    stp_result = st.session_state["stp_result"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SHOW STP/NSTP SUMMARY (read-only)
# ─────────────────────────────────────────────────────────────────────────────
if stp_result:
    st.divider()
    st.header("📋 STP / NSTP Decision Summary")

    decision   = stp_result.get("decision", "N/A")
    confidence = stp_result.get("confidence", "N/A")
    summary    = stp_result.get("customersummary", {})
    flags      = stp_result.get("reviewflags", {})

    if decision == "STP":
        st.success(f"✅ **STP** — Straight-Through Processing  |  Confidence: {confidence}")
    else:
        st.error(f"❌ **NSTP** — Manual Review Required  |  Confidence: {confidence}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Proposal No", summary.get("proposalno", "—"))
    col2.metric("Customer",    summary.get("fullname",   "—"))
    col3.metric("Occupation",  summary.get("occupation", "—"))
    col4.metric("Sum Insured", f"LKR {int(summary.get('suminsured') or 0):,}")

    # Review flags
    st.markdown("#### 🚦 Review Flags")
    fc1, fc2, fc3, fc4 = st.columns(4)
    for col, label, key in [
        (fc1, "Document Review",  "documentreviewrequired"),
        (fc2, "Medical Review",   "medicalreviewrequired"),
        (fc3, "Human Review",     "humanreviewrequired"),
        (fc4, "Loading Review",   "loadingreviewrequired"),
    ]:
        if flags.get(key):
            col.error(f"🔴 {label}")
        else:
            col.success(f"🟢 {label}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📌 Main Reasons")
        for r in stp_result.get("mainreasons", []):
            st.markdown(f"- {r}")
    with col_right:
        violated = stp_result.get("violatedrules", [])
        if violated:
            st.markdown("#### ⚠️ Violated Rules")
            st.dataframe(pd.DataFrame(violated), use_container_width=True, hide_index=True)

    with st.expander("🔍 Raw STP/NSTP JSON"):
        st.json(stp_result)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — RUN LOADING AGENT
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Premium Loading Assessment")

    if not flags.get("loadingreviewrequired"):
        st.info("ℹ️ This proposal was **not** flagged for loading review. No loading agent run required.")
    else:
        if st.button("▶️ Run Loading Agent", type="primary", use_container_width=True):
            with st.spinner("📊 Running Loading Agent..."):
                loading_result = run_loading_agent(stp_result)
                st.session_state["loading_result"] = loading_result

    # ── Display loading result if available ───────────────────────────────────
    if "loading_result" in st.session_state:
        lr          = st.session_state["loading_result"]
        ref         = lr.get("proposalreference", {})
        assignments = lr.get("loadingassignments", [])
        total_pct   = lr.get("totalloadingpercentage", 0)
        cap_applied = lr.get("capapplied", False)
        decline     = lr.get("declinerecommendation", {})
        audit       = lr.get("audittrail", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Proposal No",   ref.get("proposalno", "—"))
        col2.metric("Total Loading", f"{total_pct}%")
        col3.metric("Cap Applied",   "Yes" if cap_applied else "No")
        col4.metric("Decline",       "🔴 YES" if decline.get("shoulddecline") else "🟢 NO")

        if cap_applied:
            st.warning(f"⚠️ {lr.get('capdetails', 'Loading cap was applied.')}")
        if decline.get("shoulddecline"):
            st.error(f"🚫 **Decline Recommended:** {decline.get('reason', '')}")

        st.markdown("#### 💰 Loading Assignments")
        if assignments:
            rows = []
            for a in assignments:
                pct = a.get("loadingpercentage")
                rows.append({
                    "Loading Type"    : a.get("loadingtype", ""),
                    "Loading %"       : f"{pct}%" if pct is not None else "—",
                    "Flat Extra/1000" : a.get("flatextraper1000") or "—",
                    "Confidence"      : a.get("confidence", ""),
                    "Matched Rule"    : a.get("matchedruleid") or "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### 🔎 Detailed Breakdown")
            for a in assignments:
                lt   = a.get("loadingtype", "Unknown")
                pct  = a.get("loadingpercentage")
                conf = a.get("confidence", "LOW")
                icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
                with st.expander(f"{icon} {lt} — {pct}%  |  Confidence: {conf}"):
                    st.markdown(f"**Reason:** {a.get('reason', '—')}")
                    st.markdown(f"**Matched Rule:** `{a.get('matchedruleid', '—')}` — {a.get('matchedruledescription', '')}")
                    ev = a.get("supportingevidence", {})
                    if ev.get("questionnaireitems"):
                        st.markdown("**Questionnaire Evidence:**")
                        st.dataframe(pd.DataFrame(ev["questionnaireitems"]), use_container_width=True, hide_index=True)
                    for vd in ev.get("vectordocs", []):
                        st.markdown(f"> *[{vd.get('sourcetype')} — {vd.get('sourcename')}]* {vd.get('snippet', '')}")
        else:
            st.info("No loading assignments returned.")

        note = lr.get("importantnote", "")
        if note:
            st.info(f"📝 {note}")

        st.markdown("#### 🔒 Audit Trail")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Model:** `{audit.get('modelused', '—')}`")
        col2.markdown(f"**Version:** `{audit.get('version', '—')}`")
        col3.markdown(f"**Timestamp:** `{audit.get('timestamp', '—')}`")
        st.markdown(f"**Prompt Hash:** `{audit.get('prompthash', '—')}`")

        with st.expander("🔍 Raw Loading JSON"):
            st.json(lr)