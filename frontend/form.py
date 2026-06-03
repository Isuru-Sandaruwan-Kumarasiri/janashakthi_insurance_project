

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import requests

def main():
    st.set_page_config(page_title="Janashakthi Life Insurance Proposal", 
    page_icon="./images/logo.jpg", 
    layout="wide")
    
    # Custom CSS for a premium look
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            background-color: #2c5364;
            color: white;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #0f2027;
            color: white;
            border: 1px solid #2c5364;
        }
        .stDownloadButton>button {
            background-color: #28a745;
            color: white;
        }
        .stDownloadButton>button:hover {
            background-color: #218838;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("./images/logo.jpg", width=190)
    with col_title:
        st.markdown('<div class="main-header"><h1>Janashakthi Life Insurance Proposal Form</h1><p>Please complete using BLOCK CAPITALS. Do not use correction fluid.</p></div>', unsafe_allow_html=True)
    data = {}

    with st.expander("Administrative & Agent Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        data["proposal_no"] = col1.text_input("PROPOSAL NO:")
        data["policy_no"] = col2.text_input("POLICY NO:")
        data["agent_name"] = col3.text_input("Agent's Name:")
        data["agent_code"] = col1.text_input("Code No:")
        data["agent_contact"] = col2.text_input("Contact No:")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Personal Details", 
        "2. Dependents & Nominees", 
        "3. Plan & Benefits", 
        "4. Health & Habits", 
        "5. Medical Questionnaire", 
        "6. Verification & Submit"
    ])

    with tab1:
        st.header("1. Life to be Assured (Main Life)")
        col1, col2 = st.columns(2)
        with col1:
            min_date = datetime.today().date() - timedelta(days=100*365.25)
            max_date = datetime.today().date()
            main_life = {
                "full_name": st.text_input("1.1 Full Name (Main Life)"),
                "name_with_initials": st.text_input("1.2 Name with initials"),
                "marital_status": st.radio("1.3 Marital Status", ["Married", "Single"]),
                "dob": str(st.date_input("1.6 Date of Birth", min_value=min_date, max_value=max_date)),
                "age_next_birthday": st.number_input("Age at next birthday", min_value=0, step=1),
                "nic": st.text_input("NIC No (mandatory)")
            }
        with col2:
            main_life.update({
                "correspondence_address": st.text_area("1.4 Address for Correspondence"),
                "delivery_address": st.text_area("Delivery address (if different)"),
                "land_phone": st.text_input("1.5 Telephone (Land)"),
                "mobile_phone": st.text_input("1.5 Telephone (Mobile)"),
                "email": st.text_input("E-mail"),
                "occupation": st.text_input("1.7 Occupation & Nature of duties"),
            })
        data["main_life"] = main_life

        st.divider()
        st.header("2. Spouse Details")
        include_spouse = st.checkbox("Include Spouse in Proposal?")
        if include_spouse:
            col3, col4 = st.columns(2)
            with col3:
                min_date_spouse = datetime.today().date() - timedelta(days=100*365.25)
                max_date_spouse = datetime.today().date()
                spouse = {
                    "full_name": st.text_input("2.1 Full Name (Spouse)"),
                    "name_with_initials": st.text_input("2.2 Name with initials (Spouse)"),
                    "marital_status": st.radio("2.3 Marital Status (Spouse)", ["Married", "Single"]),
                    "dob": str(st.date_input("2.6 Date of Birth (Spouse)", min_value=min_date_spouse, max_value=max_date_spouse)),
                    "age_next_birthday": st.number_input("Age at next birthday (Spouse)", min_value=0, step=1),
                    "nic": st.text_input("NIC No (Spouse)")
                }
            with col4:
                spouse.update({
                    "correspondence_address": st.text_area("2.4 Address for Correspondence (Spouse)"),
                    "land_phone": st.text_input("2.5 Telephone (Land - Spouse)"),
                    "mobile_phone": st.text_input("2.5 Telephone (Mobile - Spouse)"),
                    "email": st.text_input("E-mail (Spouse)"),
                    "occupation": st.text_input("2.7 Occupation & Nature of duties (Spouse)"),
                    "monthly_income": st.number_input("Monthly Income Rs. (Spouse)", min_value=0.0)
                })
            data["spouse"] = spouse
        else:
            data["spouse"] = None

    with tab2:
        st.header("3. Children Details")
        st.caption("Complete only if you require Hospital and/or Critical Illness covers for children.")
        children_df = pd.DataFrame([{"Child": i+1, "Name in Full": "", "Date of Birth": "", "Gender": ""} for i in range(3)])
        edited_children = st.data_editor(children_df, use_container_width=True, hide_index=True)
        data["children"] = edited_children.to_dict(orient="records")

        st.header("4. Nominee/s Details")
        nominees_df = pd.DataFrame([{"Name in Full": "", "NIC or DOB": "", "Relationship": "", "Percentage (%)": 0} for i in range(4)])
        edited_nominees = st.data_editor(nominees_df, use_container_width=True, hide_index=True)
        data["nominees"] = edited_nominees.to_dict(orient="records")

    with tab3:
        st.header("5. Plan Type")
        plan_types = [
            "Janashakthi Cash Advance", "Janashakthi Jeevitha Varhana", 
            "Shilpashakthi", "Swarnashakthi", "Life Investment II", 
            "Life Saver", "Janashakthi Life Unlimited", "Suwashakthi"
        ]
        data["plan_type"] = st.selectbox("Select Plan Type", plan_types)

        st.header("6. Benefits")
        benefits_list = [
            "Sum Assured/Contribution (Rs.)", "Personal Accident Cover (Rs.)", 
            "Additional Life Cover (Rs.)", "Critical Illness Cover (Rs.)", 
            "Hospitalization-Daily Benefit (Rs.)", "Hospitalization-Reimbursement (Rs.)", 
            "Family Income Benefit (Rs.)", "Funeral Expenses (Rs.)"
        ]
        benefits_df = pd.DataFrame([{"Benefit": b, "Main Life": 0, "Spouse": 0, "Child 1": 0, "Child 2": 0, "Child 3": 0} for b in benefits_list])
        edited_benefits = st.data_editor(benefits_df, use_container_width=True, hide_index=True)
        data["benefits_matrix"] = edited_benefits.to_dict(orient="records")
        data["hospitalization_period"] = st.radio("Period of Hospitalization benefit", ["During the policy term & after maturity", "Only after maturity (Benefit doubled)"])

        st.header("7. Premium Details")
        col1, col2, col3, col4 = st.columns(4)
        data["premium"] = {
            "period_years": col1.number_input("Period of Policy (Years)", min_value=1),
            "frequency": col2.selectbox("Frequency", ["Mly", "Qly", "Hly", "Yly", "Single"]),
            "method": col3.selectbox("Method", ["Direct", "Salary", "Bank SO", "Other"]),
            "amount": col4.number_input("Premium Amount (Rs.)", min_value=0.0)
        }

    with tab4:
        st.header("8. Previous & Current Life Insurance Details")
        prev_insurance_df = pd.DataFrame([{"Life Assured": "Main Life", "Company Name": "", "Policy No": "", "Issue Date": "", "Total Sum Insured": 0, "Current Status": ""},
                                          {"Life Assured": "Spouse", "Company Name": "", "Policy No": "", "Issue Date": "", "Total Sum Insured": 0, "Current Status": ""}])
        data["previous_insurance"] = st.data_editor(prev_insurance_df, use_container_width=True, hide_index=True).to_dict(orient="records")

        st.header("9. Height & Weight")
        col1, col2 = st.columns(2)
        data["physical_metrics"] = {
            "main_life_height": col1.text_input("Main Life Height (cms/ins)"),
            "main_life_weight": col1.text_input("Main Life Weight (kgs/lbs)"),
            "spouse_height": col2.text_input("Spouse Height (cms/ins)"),
            "spouse_weight": col2.text_input("Spouse Weight (kgs/lbs)")
        }

        st.header("10. Habits (Smoking & Drinking)")
        habits_df = pd.DataFrame([
            {"Person": "Main Life (Smoke)", "Yes/No": "No", "Type": "", "Qty/Week": "", "How Long (Yrs)": ""},
            {"Person": "Main Life (Alcohol)", "Yes/No": "No", "Type": "", "Qty/Week": "", "How Long (Yrs)": ""},
            {"Person": "Spouse (Smoke)", "Yes/No": "No", "Type": "", "Qty/Week": "", "How Long (Yrs)": ""},
            {"Person": "Spouse (Alcohol)", "Yes/No": "No", "Type": "", "Qty/Week": "", "How Long (Yrs)": ""}
        ])
        data["habits"] = st.data_editor(habits_df, use_container_width=True, hide_index=True).to_dict(orient="records")

        st.header("11. Family History")
        family_members = ["Father", "Mother", "Brother/s", "Sister/s", "Spouse"]
        family_df = pd.DataFrame([{"Person": m, "Living? (Yes/No)": "Yes", "Present Health": "", "Age at Death": "", "Cause of Death": ""} for m in family_members])
        st.write("Main Life Family History")
        data["family_history_main"] = st.data_editor(family_df, key="fam_main", use_container_width=True, hide_index=True).to_dict(orient="records")
        st.write("Spouse Family History")
        data["family_history_spouse"] = st.data_editor(family_df, key="fam_spouse", use_container_width=True, hide_index=True).to_dict(orient="records")

    with tab5:
        st.header("12. Medical History")
        med_questions = [
            "1. Visited a doctor in the last 3 years?",
            "2. Subject to any medical condition, illness or injury?",
            "3. Undergone or advised to undergo an operation/X-ray?",
            "4. Illness, accident or injury requiring overnight hospital stay?",
            "5. At present receiving medical treatment or taking medicine?",
            "6. State of anxiety, depression or mental/neurological disorder?",
            "7. Physically disabled or defect due to injury/disease?",
            "8. Physical disability, deformity or impairment?",
            "9. Absent from work on grounds of ill health (last 3 yrs for >3 days)?",
            "10a. Ailments of heart, circulatory, high blood pressure, stroke?",
            "10b. Diabetes, kidney, eyes, ears ailments?",
            "10c. Ailments of brain, central nervous system, mental illness?",
            "10d. Cancer, cyst, tumour or blood cancers?",
            "10e. Multiple sclerosis, arthritis, rheumatism?",
            "10f. Hepatitis, AIDS or AIDS related condition?",
            "10g. Respiratory/lung disease (Asthma, bronchitis)?",
            "10h. Ailments of digestive system, liver, ulcer, hernia?",
            "11. Any other illness or disorder?"
        ]
        
        med_responses = {}
        for q in med_questions:
            med_responses[q] = st.checkbox(q)
        
        data["medical_history"] = med_responses
        data["medical_details"] = st.text_area("If any answered 'Yes', please provide more details:")

        st.header("13. Additional Questions")
        data["additional_questions"] = {
            "hazardous_occupation": st.checkbox("1. Intention of engaging in hazardous occupation?"),
            "hazardous_sport": st.checkbox("2. Engage in hazardous sport (racing, diving)?"),
            "criminal_offense": st.checkbox("3. Arrested/convicted of criminal offense?"),
            "life_threat": st.checkbox("4. Threat on your/family's lives?"),
            "details": st.text_area("If 'Yes' to any above, give details:")
        }

        st.header("14. Specific Questions for Females")
        data["female_questions"] = {
            "pregnant": st.checkbox("1. Pregnant at present?"),
            "last_menstruation": str(st.date_input("Last date of menstruation (if pregnant)")),
            "complications": st.checkbox("2. Advised/treated for pregnancy complications?"),
            "disorders": st.checkbox("3. Disorder of female organs (breasts, ovaries, uterus)?"),
            "details": st.text_area("If 'Yes' to female questions, give details:")
        }

    with tab6:
        st.header("15. Preferred Language")
        data["language"] = st.radio("Preferred Language for policy document", ["Sinhala", "Tamil", "English"])

        st.header("Initial Deposit Details")
        col1, col2 = st.columns(2)
        data["deposit"] = {
            "installment_premium": col1.number_input("Installment premium (Rs.)", min_value=0.0),
            "initial_deposit": col2.number_input("Initial deposit amount (Rs.)", min_value=0.0),
            "payment_method": col1.radio("Payment Method", ["Cash", "Cheque"]),
            "date_of_payment": str(col2.date_input("Date of payment")),
            "cheque_no": col1.text_input("Cheque No."),
            "bank": col2.text_input("Bank")
        }

        st.header("16. Declaration")
        st.markdown("I/We DECLARE with best of my/our knowledge as at date the statements made in this proposal are true...")
        declaration_agreed = st.checkbox("I agree to the Declaration")
        
        st.divider()
        
        st.header("17. Document Verification & Submission")
        st.markdown("""
        Upload your **ID Card** (NIC/Passport) and a **Utility Bill** (water/electricity bill).
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🪪 ID Card Upload")
            id_card_file = st.file_uploader(
                "Upload your NIC / Passport / ID Card",
                type=["jpg", "jpeg", "png", "tiff", "bmp", "webp", "pdf"],
                key="id_card_upload"
            )
            if id_card_file:
                st.success(f"Uploaded: {id_card_file.name}")
                if id_card_file.type and id_card_file.type.startswith("image"):
                    st.image(id_card_file, caption="ID Card Preview", use_container_width=True)

        with col2:
            st.subheader("💧 Utility Bill Upload")
            utility_bill_file = st.file_uploader(
                "Upload your Water / Electricity Bill",
                type=["jpg", "jpeg", "png", "tiff", "bmp", "webp", "pdf"],
                key="utility_bill_upload"
            )
            if utility_bill_file:
                st.success(f"Uploaded: {utility_bill_file.name}")
                if utility_bill_file.type and utility_bill_file.type.startswith("image"):
                    st.image(utility_bill_file, caption="Utility Bill Preview", use_container_width=True)

        st.divider()
        
        if "verification_result" not in st.session_state:
            st.session_state.verification_result = None

        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            verify_clicked = st.button("🔍 Step 1: Verify Documents", use_container_width=True)
            
        if verify_clicked:
            if not declaration_agreed:
                st.error("You must agree to the declaration before verifying.")
            elif not id_card_file:
                st.error("Please upload an ID card image.")
            elif not utility_bill_file:
                st.error("Please upload a utility bill image.")
            else:
                with st.spinner("Running OCR and AI Document Verification... (This may take 30-60 seconds)"):
                    # Reset file pointers to 0 in case they were read by st.image
                    id_card_file.seek(0)
                    utility_bill_file.seek(0)
                    
                    try:
                        files = {
                            "id_card_image": (id_card_file.name, id_card_file.read(), id_card_file.type),
                            "utility_bill_image": (utility_bill_file.name, utility_bill_file.read(), utility_bill_file.type),
                        }
                        form_data = {"proposal_data": json.dumps(data)}
                        
                        ver_response = requests.post(
                            "http://localhost:8000/api/verify-documents",
                            files=files,
                            data=form_data,
                            timeout=300
                        )

                        if ver_response.status_code == 200:
                            st.session_state.verification_result = ver_response.json()
                        else:
                            st.error(f"Verification failed. Status Code: {ver_response.status_code}")
                            st.error(ver_response.json().get("detail", ver_response.text))
                            st.session_state.verification_result = None
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the backend API. Make sure the FastAPI server is running on port 8000.")
                    except requests.exceptions.Timeout:
                        st.error("Verification timed out. The LangGraph process took too long.")

        if st.session_state.verification_result:
            vr = st.session_state.verification_result
            score = vr.get("overall_score", 0)
            overall = vr.get("overall_status", "UNKNOWN")
            
            st.divider()
            
            if score >= 80:
                st.success(f"✅ Document Verification Score: **{score}/100** — {overall}")
            else:
                st.error(f"❌ Document Verification Score: **{score}/100** — {overall}")
            st.progress(min(score, 100) / 100)
            st.info(vr.get("summary", ""))
            
            checks = vr.get("checks", [])
            failed_fields = []
            for check in checks:
                is_match = check.get("match", False)
                match_icon = "✅" if is_match else "❌"
                field_score = check.get("score", 0)
                max_score = check.get("max_score", 0)
                field_name = check.get("field", "Unknown")
                
                if not is_match:
                    failed_fields.append(field_name)
                
                with st.expander(f"{match_icon} {field_name} — {field_score}/{max_score} pts"):
                    ca, cb = st.columns(2)
                    ca.metric("Proposal Value", check.get("proposal_value", "N/A"))
                    cb.metric("Document Value", check.get("document_value", "N/A"))
                    st.write(f"**Reasoning:** {check.get('reasoning', '')}")
            
            with st.expander("🔎 Raw OCR: ID Card Data"):
                st.json(vr.get("id_card_ocr", {}))
            with st.expander("🔎 Raw OCR: Utility Bill Data"):
                st.json(vr.get("utility_bill_ocr", {}))

            with col_btn2:
                if score >= 80:
                    submit_clicked = st.button("💾 Step 2: Submit Proposal to S3", type="primary", use_container_width=True)
                    if submit_clicked:
                        data["declaration_agreed"] = True
                        data["document_verification_result"] = vr
                        
                        proposal_id = data.get("proposal_no", "").strip() or data.get("policy_no", "").strip()
                        
                        if not proposal_id:
                            st.error("PROPOSAL NO or POLICY NO is required to save the record to S3.")
                        else:
                            with st.spinner("Saving Proposal Data to AWS S3..."):
                                try:
                                    api_url = f"http://localhost:8000/api/proposals/{proposal_id}"
                                    s3_response = requests.post(api_url, json=data)
                                    
                                    if s3_response.status_code == 200:
                                        st.success(f"🎉 Proposal {proposal_id} successfully saved to AWS S3!")
                                        with st.expander("View Full Submitted Data"):
                                            st.json(data)
                                    else:
                                        st.error(f"Failed to save to S3. Status Code: {s3_response.status_code}")
                                        st.error(s3_response.text)
                                except Exception as e:
                                    st.error(f"Failed to save to S3: {e}")
                else:
                    if failed_fields:
                        st.error(f"❌ Cannot submit. Score is {score}% (minimum 80% required). The following fields did NOT match: **{', '.join(failed_fields)}**.")
                    else:
                        st.error(f"❌ Cannot submit. Score is {score}% (minimum 80% required).")
                    st.button("💾 Step 2: Submit Proposal to S3", disabled=True, use_container_width=True)

if __name__ == "__main__":
    main()
