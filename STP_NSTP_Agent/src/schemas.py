# =========================================================
# src/schemas.py
# Input and output schemas for LLM-first STP/NSTP Rule RAG Agent
#
# OCR + Verification Agent already checks identity/address/NIC/DOB.
# This agent receives only verified proposal JSON.
# =========================================================

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ProposalMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    proposalno: Optional[str] = None
    policy_no: Optional[str] = None
    source: Optional[str] = "verified_frontend_proposal"
    form_type: Optional[str] = "Janashakthi Proposal for Life Insurance"


class CustomerContact(BaseModel):
    model_config = ConfigDict(extra="allow")

    full_name: Optional[str] = None
    name_with_initials: Optional[str] = None
    telephone_land: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    preferred_language: Optional[str] = None


class PersonalDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    age: Optional[Any] = None
    date_of_birth: Optional[str] = None
    nic: Optional[str] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    address: Optional[str] = None
    occupation: Optional[str] = None
    company_or_institution: Optional[str] = None
    monthly_income: Optional[Any] = None


class ProposalDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    plan_type: Optional[str] = None
    sum_insured: Optional[Any] = None
    premium_amount: Optional[Any] = None
    payment_frequency: Optional[str] = None
    payment_method: Optional[str] = None
    riders: List[Any] = Field(default_factory=list)


class PreviousInsuranceDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    has_existing_or_previous_policy: Optional[str] = "No"
    has_declined_or_postponed_policy: Optional[str] = "No"
    details: List[Any] = Field(default_factory=list)


class PhysicalDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    height_cm: Optional[Any] = None
    weight_kg: Optional[Any] = None
    bmi: Optional[Any] = None


class Habits(BaseModel):
    model_config = ConfigDict(extra="allow")

    smoker: Optional[str] = "No"
    smoking_quantity_per_week: Optional[Any] = None
    smoking_duration_years: Optional[Any] = None
    alcohol: Optional[str] = "No"
    alcohol_quantity_ml: Optional[Any] = None
    alcohol_duration_years: Optional[Any] = None


class FamilyHistory(BaseModel):
    model_config = ConfigDict(extra="allow")

    has_family_medical_history: Optional[str] = "No"
    family_disease_details: List[Any] = Field(default_factory=list)


class SpecificDiseaseHistory(BaseModel):
    model_config = ConfigDict(extra="allow")

    heart_or_blood_pressure_or_stroke: Optional[str] = "No"
    kidney_or_urinary_or_eyes_ears: Optional[str] = "No"
    brain_or_nervous_or_mental: Optional[str] = "No"
    cancer_or_tumour: Optional[str] = "No"
    multiple_sclerosis_or_arthritis_or_rheumatism: Optional[str] = "No"
    hepatitis_aids_or_related_condition: Optional[str] = "No"
    respiratory_or_asthma_or_bronchitis: Optional[str] = "No"
    digestive_gall_bladder_liver_ulcer_bleeding: Optional[str] = "No"


class MedicalHistory(BaseModel):
    model_config = ConfigDict(extra="allow")

    visited_doctor_last_3_years: Optional[str] = "No"
    had_medical_condition_or_injury: Optional[str] = "No"
    had_operation_xray_or_hospital_test: Optional[str] = "No"
    overnight_hospital_stay: Optional[str] = "No"
    currently_taking_treatment_or_drugs: Optional[str] = "No"
    mental_or_neurological_condition: Optional[str] = "No"
    physical_disability_or_defect: Optional[str] = "No"
    physical_disability_deformity_or_impairment: Optional[str] = "No"
    absent_from_work_due_to_health: Optional[str] = "No"
    specific_disease_history: SpecificDiseaseHistory = Field(default_factory=SpecificDiseaseHistory)
    medical_details_text: Optional[str] = None


class AdditionalQuestions(BaseModel):
    model_config = ConfigDict(extra="allow")

    hazardous_occupation: Optional[str] = "No"
    hazardous_sport: Optional[str] = "No"
    criminal_offence: Optional[str] = "No"
    threat_on_life: Optional[str] = "No"


class FemaleQuestions(BaseModel):
    model_config = ConfigDict(extra="allow")

    pregnant_present: Optional[str] = None
    pregnancy_complication_history: Optional[str] = None
    female_organ_disorder: Optional[str] = None


class BranchOfficeChecks(BaseModel):
    model_config = ConfigDict(extra="allow")

    q11_family_question_yes: Optional[str] = "No"
    q12_medical_question_yes: Optional[str] = "No"
    q13_additional_question_yes: Optional[str] = "No"
    q14_female_question_yes: Optional[str] = "No"


class VerificationResult(BaseModel):
    """
    Optional only. OCR agent already blocks unverified submissions.
    This agent does not require completeness/authenticity scores.
    """
    model_config = ConfigDict(extra="allow")

    identity_verified: Optional[bool] = True
    address_verified: Optional[bool] = True
    nic_verified: Optional[bool] = True
    dob_verified: Optional[bool] = True
    mismatches: List[Any] = Field(default_factory=list)
    missing_documents: List[Any] = Field(default_factory=list)


class ProposalInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    proposal_metadata: ProposalMetadata = Field(default_factory=ProposalMetadata)
    customer_contact: CustomerContact = Field(default_factory=CustomerContact)
    personal_details: PersonalDetails = Field(default_factory=PersonalDetails)
    proposal_details: ProposalDetails = Field(default_factory=ProposalDetails)
    previous_insurance_details: PreviousInsuranceDetails = Field(default_factory=PreviousInsuranceDetails)
    physical_details: PhysicalDetails = Field(default_factory=PhysicalDetails)
    habits: Habits = Field(default_factory=Habits)
    family_history: FamilyHistory = Field(default_factory=FamilyHistory)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    additional_questions: AdditionalQuestions = Field(default_factory=AdditionalQuestions)
    female_questions: FemaleQuestions = Field(default_factory=FemaleQuestions)
    branch_office_checks: BranchOfficeChecks = Field(default_factory=BranchOfficeChecks)
    verification_result: VerificationResult = Field(default_factory=VerificationResult)


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    proposalno: Optional[str] = None
    customer_summary: Dict[str, Any] = Field(default_factory=dict)

    decision: str = "NSTP"
    stp_eligible: bool = False
    confidence: str = "LOW"

    violated_rules: List[Dict[str, Any]] = Field(default_factory=list)
    rule_check_summary: Dict[str, Any] = Field(default_factory=dict)

    required_documents: List[str] = Field(default_factory=list)
    required_medical_reports: List[str] = Field(default_factory=list)
    document_recommendation_reasons: List[str] = Field(default_factory=list)

    retrieval_summary: Dict[str, Any] = Field(default_factory=dict)
    sql_past_case_context: Dict[str, Any] = Field(default_factory=dict)
    vector_past_case_context: Dict[str, Any] = Field(default_factory=dict)

    loading_agent_input: Dict[str, Any] = Field(default_factory=dict)
    gui_summary: Dict[str, Any] = Field(default_factory=dict)

    agent_process_status: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None


def validate_proposal_input(proposal_json: Dict[str, Any]) -> ProposalInput:
    return ProposalInput.model_validate(proposal_json)


def proposal_input_to_dict(proposal_input: ProposalInput) -> Dict[str, Any]:
    return proposal_input.model_dump()


def validate_agent_output(output_json: Dict[str, Any]) -> AgentOutput:
    return AgentOutput.model_validate(output_json)


def agent_output_to_dict(agent_output: AgentOutput) -> Dict[str, Any]:
    return agent_output.model_dump()


def build_error_output(
    proposal_json: Optional[Dict[str, Any]] = None,
    error_message: str = "Unknown error"
) -> Dict[str, Any]:
    proposalno = None
    if isinstance(proposal_json, dict):
        proposalno = proposal_json.get("proposal_metadata", {}).get("proposalno")

    return AgentOutput(
        proposalno=proposalno,
        decision="NSTP",
        stp_eligible=False,
        confidence="LOW",
        violated_rules=[],
        rule_check_summary={
            "summary": "Agent failed safely and routed proposal to NSTP.",
            "reason": "System error"
        },
        required_documents=[],
        required_medical_reports=[],
        document_recommendation_reasons=[
            "System error occurred before document recommendation could be completed."
        ],
        retrieval_summary={
            "sql_tool_used": False,
            "vector_retrieval_used": False,
            "similar_sql_cases_count": 0,
            "vector_docs_count": 0
        },
        loading_agent_input={
            "should_send_to_loading_agent": True,
            "reason": "Agent failed safely, so downstream review may be required.",
            "proposal_json": proposal_json or {},
            "violated_rules": [],
            "required_documents": [],
            "required_medical_reports": []
        },
        gui_summary={
            "status_badge": "NSTP",
            "display_message": "Proposal requires review due to an internal agent error.",
            "next_step": "Send proposal to human underwriter or rerun after fixing the error.",
            "highlight_items": ["System error"]
        },
        agent_process_status={
            "rule_check_completed": False,
            "sql_retrieval_completed": False,
            "vector_retrieval_completed": False,
            "final_document_analysis_completed": False
        },
        status="error",
        error=error_message
    ).model_dump()
