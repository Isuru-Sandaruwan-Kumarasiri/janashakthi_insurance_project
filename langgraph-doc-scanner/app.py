"""
app.py — Streamlit-based web frontend for the Document Processing Agent.

Allows users to upload document images or PDFs, configure model settings,
run the unified LangGraph processing pipeline, and view the structured JSON
output with a beautiful dashboard view.

Usage:
    streamlit run OCR/app.py
"""

from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import pandas as pd
from PIL import Image

# Ensure notebooks directory is in python path
notebooks_dir = Path(__file__).parent.absolute()
if str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

from ocr_agent.document_agent import process_document
from core.schemas import DocumentType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import fitz for PDF rendering
try:
    import fitz  # PyMuPDF
    fitz_available = True
except ImportError:
    fitz_available = False
    logger.warning("PyMuPDF (fitz) is not installed. PDF page previews will be disabled.")


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocAgent Hub — Intelligent Document Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism, nice badges and cards
st.markdown("""
<style>
    /* Gradient Header styling */
    .main-title {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #555566;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Dark mode adjustments for subtitles */
    @media (prefers-color-scheme: dark) {
        .subtitle {
            color: #aaaabb;
        }
    }

    /* Cards styling */
    .info-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.25rem;
    }

    /* Custom badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
    }
    .badge-danger {
        background-color: #dc3545;
        color: white;
    }
    .badge-info {
        background-color: #17a2b8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Helper function to save uploaded file
def save_uploaded_file(uploaded_file) -> Path:
    temp_dir = Path(__file__).parent.parent / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Preserve suffix
    suffix = Path(uploaded_file.name).suffix
    temp_path = temp_dir / f"upload_{uploaded_file.file_id}{suffix}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return temp_path


# ---------------------------------------------------------
# Sidebar - Configuration
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/200/document.png", width=120)
st.sidebar.title("DocAgent Settings")

# Model override selection
model_options = {
    "openai/gpt-4o-mini": "GPT-4o Mini (Default)",
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "meta-llama/llama-3-70b-instruct": "Llama 3 70B",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash"
}
selected_model_key = st.sidebar.selectbox(
    "LLM Extraction Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

max_retries = st.sidebar.slider(
    "Max Retries (on validation failure)",
    min_value=0,
    max_value=4,
    value=2
)

# API key validator
st.sidebar.markdown("---")
st.sidebar.subheader("Environment Keys")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key (optional override)",
    type="password",
    help="Leave blank to use OPENROUTER_API_KEY from the project .env file."
)


# Set env variables before process execution
os.environ["OPENROUTER_MODEL"] = selected_model_key
if openrouter_api_key:
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>📄 DocAgent Hub</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Unified OCR, Document Classification, LLM Extraction, and Schema Validation Pipeline</p>",
    unsafe_allow_html=True
)

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("### 1. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "pdf"],
        help="Upload a utility bill, ID card, or medical report to extract data."
    )
    
    temp_file_path = None
    page_count = 1
    selected_page = 0
    
    if uploaded_file is not None:
        temp_file_path = save_uploaded_file(uploaded_file)
        
        # Check PDF page count
        if temp_file_path.suffix.lower() == ".pdf":
            if fitz_available:
                try:
                    doc = fitz.open(temp_file_path)
                    page_count = len(doc)
                    doc.close()
                    
                    if page_count > 1:
                        selected_page = st.number_input(
                            f"Select Page (Document has {page_count} pages)",
                            min_value=1,
                            max_value=page_count,
                            value=1,
                            help="Specify which page to process (1-indexed)."
                        ) - 1
                except Exception as e:
                    st.error(f"Error reading PDF page structure: {e}")
            else:
                st.warning("PyMuPDF is not installed. Defaulting to first page.")
        
        # Action button
        st.markdown("### 2. Process Document")
        run_btn = st.button("🚀 Run Extraction Pipeline", type="primary", use_container_width=True)
        
        if run_btn:
            # Verification of API Key
            api_key_check = os.getenv("OPENROUTER_API_KEY") or openrouter_api_key
            if not api_key_check:
                st.error("❌ OPENROUTER_API_KEY is not set. Please add it to your .env file or enter it in the sidebar.")
            else:
                with st.spinner("Processing document through pipeline (Preprocessing -> OCR -> Classify -> Extract -> Validate)..."):
                    try:
                        start_time = datetime.now()
                        
                        # Execute the pipeline
                        result = process_document(
                            input_path=str(temp_file_path),
                            page_number=selected_page,
                            max_retries=max_retries
                        )
                        
                        # Store in session state
                        st.session_state["pipeline_result"] = result
                        st.session_state["pipeline_success"] = True
                        st.session_state["processing_duration"] = (datetime.now() - start_time).total_seconds()
                        
                        st.success(f"✅ Pipeline completed in {st.session_state['processing_duration']:.2f} seconds!")
                    except Exception as e:
                        st.error(f"Pipeline Execution Failed: {e}")
                        st.exception(e)

with col_preview:
    st.markdown("### Document Preview")
    if uploaded_file is not None and temp_file_path is not None:
        if temp_file_path.suffix.lower() == ".pdf":
            if fitz_available:
                try:
                    doc = fitz.open(temp_file_path)
                    page = doc.load_page(selected_page)
                    # Render page at 150 DPI for UI preview
                    pix = page.get_pixmap(matrix=fitz.Matrix(150/72.0, 150/72.0))
                    img_bytes = pix.tobytes("png")
                    st.image(img_bytes, caption=f"PDF Page {selected_page + 1} of {page_count}", use_container_width=True)
                    doc.close()
                except Exception as e:
                    st.error(f"Could not render PDF preview: {e}")
            else:
                st.info("PDF document uploaded. Previews are disabled (install PyMuPDF).")
        else:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption=uploaded_file.name, use_container_width=True)
            except Exception as e:
                st.error(f"Could not display image preview: {e}")
    else:
        # Placeholder styling
        st.markdown(
            """
            <div style="border: 2px dashed rgba(128,128,128,0.3); border-radius: 12px; height: 300px; display: flex; align-items: center; justify-content: center; color: rgba(128,128,128,0.7);">
                Upload a document to preview it here
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# Results Section
# ---------------------------------------------------------
if st.session_state.get("pipeline_success", False):
    result = st.session_state["pipeline_result"]
    doc_type = result.get("document_type", "unknown")
    extracted_data = result.get("extracted_data", {})
    validation = result.get("validation", {})
    metadata = result.get("metadata", {})
    
    st.markdown("---")
    st.markdown("## 📊 Processing Results")
    
    # 4 tabs for visualization
    tab_dashboard, tab_validation, tab_json, tab_ocr = st.tabs([
        "📊 Dashboard View", 
        "🛡️ Validation & Quality", 
        "⚙️ Raw JSON Output", 
        "🔍 OCR Text & Zones"
    ])
    
    # ------------------
    # TAB 1: Dashboard View
    # ------------------
    with tab_dashboard:
        # Document Type Badge Banner
        type_banners = {
            "id_card": ("Identity Card", "badge-success"),
            "water_bill": ("Water Utility Bill", "badge-info"),
            "medical_report": ("Medical Report", "badge-warning"),
            "unknown": ("Unknown Document", "badge-danger")
        }
        type_display, badge_class = type_banners.get(doc_type, (doc_type.upper(), "badge-danger"))
        
        st.markdown(
            f"<h4>Detected Document Type: <span class='badge {badge_class}'>{type_display}</span></h4>",
            unsafe_allow_html=True
        )
        
        # Display fields dynamically based on doc_type
        if doc_type == "id_card":
            st.markdown("### Extracted Identity Information")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Full Name", value=extracted_data.get("full_name") or "", disabled=True)
                st.text_input("ID / NIC Number", value=extracted_data.get("id_number") or "", disabled=True)
                st.text_input("Date of Birth", value=extracted_data.get("date_of_birth") or "", disabled=True)
                st.text_input("Nationality", value=extracted_data.get("nationality") or "", disabled=True)
            with c2:
                st.text_input("Gender", value=extracted_data.get("gender") or "", disabled=True)
                st.text_input("Issue Date", value=extracted_data.get("issue_date") or "", disabled=True)
                st.text_input("Expiry Date", value=extracted_data.get("expiry_date") or "", disabled=True)
                st.text_area("Address", value=extracted_data.get("address") or "", height=80, disabled=True)
                
            if extracted_data.get("mrz_line1") or extracted_data.get("mrz_line2"):
                with st.expander("Machine Readable Zone (MRZ)"):
                    st.code(f"{extracted_data.get('mrz_line1') or ''}\n{extracted_data.get('mrz_line2') or ''}")

        elif doc_type == "water_bill":
            st.markdown("### Extracted Utility Information")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Customer Name", value=extracted_data.get("customer_name") or "", disabled=True)
                st.text_input("Account Number", value=extracted_data.get("account_number") or "", disabled=True)
                st.text_area("Service Address", value=extracted_data.get("service_address") or "", height=80, disabled=True)
                st.text_input("Billing Period", value=extracted_data.get("billing_period") or "", disabled=True)
            with c2:
                st.text_input("Amount Due", value=extracted_data.get("amount_due") or "", disabled=True)
                st.text_input("Due Date", value=extracted_data.get("due_date") or "", disabled=True)
                st.text_input("Previous Meter Reading", value=extracted_data.get("previous_reading") or "", disabled=True)
                st.text_input("Current Meter Reading", value=extracted_data.get("current_reading") or "", disabled=True)
                st.text_input("Consumption Units", value=extracted_data.get("consumption_units") or "", disabled=True)
                st.text_input("Tariff Rate", value=extracted_data.get("tariff_rate") or "", disabled=True)

        elif doc_type == "medical_report":
            st.markdown("### Extracted Clinical Information")
            
            # Demographics
            st.markdown("#### Patient Demographics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.text_input("Patient Name", value=extracted_data.get("patient_name") or "", disabled=True)
            with c2:
                st.text_input("Patient ID", value=extracted_data.get("patient_id") or "", disabled=True)
            with c3:
                st.text_input("Date of Birth", value=extracted_data.get("date_of_birth") or "", disabled=True)
            with c4:
                st.text_input("Gender", value=extracted_data.get("gender") or "", disabled=True)
                
            # Metadata
            st.markdown("#### Report Metadata")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.text_input("Report Date", value=extracted_data.get("report_date") or "", disabled=True)
            with c2:
                st.text_input("Report Type", value=extracted_data.get("report_type") or "", disabled=True)
            with c3:
                st.text_input("Hospital / Institution", value=extracted_data.get("hospital") or "", disabled=True)
            with c4:
                st.text_input("Ordering Physician", value=extracted_data.get("physician") or "", disabled=True)
                
            # Diagnosis, Meds, Allergies
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("<div class='info-card'><div class='card-title'>Diagnoses</div>", unsafe_allow_html=True)
                diagnoses = extracted_data.get("diagnosis", [])
                if diagnoses:
                    for d in diagnoses:
                        st.markdown(f"- **{d}**")
                else:
                    st.write("None recorded")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='info-card'><div class='card-title'>Medications</div>", unsafe_allow_html=True)
                meds = extracted_data.get("medications", [])
                if meds:
                    for m in meds:
                        st.markdown(f"- {m}")
                else:
                    st.write("None recorded")
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='info-card'><div class='card-title'>Allergies</div>", unsafe_allow_html=True)
                allergies = extracted_data.get("allergies", [])
                if allergies:
                    for a in allergies:
                        st.markdown(f"- {a}")
                else:
                    st.write("None recorded")
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Vitals
            raw_vitals = extracted_data.get("vitals")
            # vitals may be a dict (from LLM JSON) or None
            vitals = raw_vitals if isinstance(raw_vitals, dict) else {}
            if vitals and any(vitals.values()):
                with st.expander("Vital Signs", expanded=True):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("Blood Pressure", vitals.get("blood_pressure") or "—")
                    with c2:
                        st.metric("Heart Rate", vitals.get("heart_rate") or "—")
                    with c3:
                        st.metric("Temperature", vitals.get("temperature") or "—")
                    with c4:
                        st.metric("SpO2", vitals.get("oxygen_saturation") or "—")
                    with c5:
                        st.metric("Weight / Height", f"{vitals.get('weight') or '—'} / {vitals.get('height') or '—'}")
            
            # Lab Values
            lab_values_raw = extracted_data.get("lab_values", [])
            # Ensure we only pass dicts to DataFrame (guard against Pydantic model objects or strings)
            lab_values = [lv for lv in (lab_values_raw or []) if isinstance(lv, dict)]
            if lab_values:
                st.markdown("#### Laboratory Results")
                df_labs = pd.DataFrame(lab_values)
                # Fill missing keys to prevent UI issues
                for col in ["test_name", "value", "unit", "reference_range", "flag", "is_abnormal"]:
                    if col not in df_labs.columns:
                        df_labs[col] = None
                df_labs = df_labs[["test_name", "value", "unit", "reference_range", "flag", "is_abnormal"]]
                st.dataframe(df_labs, use_container_width=True)

            if extracted_data.get("clinical_notes"):
                st.text_area("Clinical Notes", value=extracted_data.get("clinical_notes"), height=100, disabled=True)
        
        else:
            st.markdown("### Generic Extracted Key-Values")
            if extracted_data:
                st.write(extracted_data)
            else:
                st.warning("No data extracted by LLM.")

    # ------------------
    # TAB 2: Validation & Quality
    # ------------------
    with tab_validation:
        st.markdown("### 🛡️ Schema Verification & Metrics")
        
        # Validation Box
        is_valid = validation.get("is_valid", False)
        val_score = validation.get("confidence_score", 0.0)
        requires_review = validation.get("requires_human_review", False)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if is_valid:
                st.success("✅ SCHEMA VALIDATION: PASSED")
            else:
                st.error("❌ SCHEMA VALIDATION: FAILED")
        with c2:
            st.metric("Overall Validation Confidence", f"{val_score * 100:.1f}%")
        with c3:
            if requires_review:
                st.warning("⚠️ Requires Human Review")
            else:
                st.success("✨ Verification Complete")
                
        # Error & Warning lists
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])
        
        if errors:
            st.markdown("#### Validation Errors")
            for err in errors:
                st.markdown(f"- 🔴 {err}")
                
        if warnings:
            st.markdown("#### Validation Warnings")
            for warn in warnings:
                st.markdown(f"- 🟡 {warn}")
                
        # Quality Metrics
        st.markdown("---")
        st.markdown("#### Quality Metrics")
        mq1, mq2, mq3, mq4 = st.columns(4)
        with mq1:
            st.metric("Image Quality Score", f"{metadata.get('image_quality_score', 0.0) * 100:.1f}%")
        with mq2:
            st.metric("Classifier Confidence", f"{metadata.get('classifier_confidence', 0.0) * 100:.1f}%")
        with mq3:
            st.metric("OCR Zones Detected", metadata.get("ocr_zones_count", 0))
        with mq4:
            st.metric("Extraction Retries", metadata.get("retry_count", 0))

    # ------------------
    # TAB 3: Raw JSON Output
    # ------------------
    with tab_json:
        st.markdown("### ⚙️ Raw Agent Output JSON")
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")
        
        # Download button
        st.download_button(
            label="💾 Download JSON Result",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"extracted_{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # ------------------
    # TAB 4: OCR Text & Zones
    # ------------------
    with tab_ocr:
        st.markdown("### 🔍 Raw Extracted OCR Text")
        st.text_area(
            "Ordered OCR Text", 
            value=st.session_state.get("pipeline_result", {}).get("metadata", {}).get("raw_ocr_text", "No OCR text in response metadata") 
            if "raw_ocr_text" in st.session_state.get("pipeline_result", {}).get("metadata", {}) 
            else "No raw OCR text available",
            height=250
        )
        
        # If we can retrieve zones from state or if we have to display list
        st.markdown("#### Detected Zones")
        # In a real environment, we'd pull this from the graph state. Since the final output doesn't contain
        # the entire raw zone coordinate list by default (only count is in metadata), we inform the user.
        st.write(f"The OCR engine detected **{metadata.get('ocr_zones_count', 0)}** semantic zones on the page, using a combination of TrOCR (handwriting) and PaddleOCR (printed).")
