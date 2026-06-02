"""
test_integration.py — Integration test for the Janashakthi Document Verification and Storage API.

Verifies that:
1. The backend is running on port 8000.
2. The OCR + LLM comparison works end-to-end on test images.
3. The proposal can be successfully saved to S3.
"""

import sys
import json
import requests
from pathlib import Path

BACKEND_URL = "http://localhost:8000"

def main():
    print("=" * 60)
    print("        Janashakthi OCR Integration Test")
    print("=" * 60)

    # 1. Check health
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            print(f"[Health] Backend is UP and running: {resp.json()}")
        else:
            print(f"[Health] Backend returned status code {resp.status_code}: {resp.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"[Error] Could not connect to backend at {BACKEND_URL}.")
        print("Please start the backend first (e.g. by running: python run_app.py)")
        sys.exit(1)

    # Define test files
    root_dir = Path(__file__).resolve().parent
    id_card_path = root_dir / "data" / "university_id.jpg"
    utility_bill_path = root_dir / "data" / "university_id.jpg" # Using same image for testing OCR pipeline

    if not id_card_path.exists():
        print(f"[Error] Test file not found at {id_card_path}")
        sys.exit(1)

    # Prepare mock proposal data matching the university ID details
    proposal_data = {
        "proposal_no": "PROP-TEST-999",
        "policy_no": "POL-TEST-999",
        "agent_name": "Test Agent",
        "main_life": {
            "full_name": "PELANDA KURUNDUKARAGE ISURU SANDARUWAN KUMARASIRI",
            "name_with_initials": "P. K. I. S. Kumarasiri",
            "marital_status": "Single",
            "dob": "2000-01-01",
            "nic": "S/20/426",
            "correspondence_address": "14/1/B, NEAR THE DAMME BRIDGE, ELAPATHA, RATHNAPU",
            "mobile_phone": "0771234567",
            "email": "test@example.com",
            "occupation": "Student"
        }
    }

    print("\n[Test 1] Running OCR + LLM comparison pipeline...")
    print("  This involves: loading PaddleOCR models, running image preprocessing,")
    print("  extracting text, calling VLM, and doing LLM comparison. (May take 1 min)...")

    files = {
        "id_card_image": ("university_id.jpg", id_card_path.read_bytes(), "image/jpeg"),
        "utility_bill_image": ("utility_bill.jpg", utility_bill_path.read_bytes(), "image/jpeg")
    }
    
    data = {
        "proposal_data": json.dumps(proposal_data)
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/verify-documents",
            files=files,
            data=data,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n[Result] Verification request completed successfully!")
            print(f"  Overall Status: {result.get('overall_status')}")
            print(f"  Overall Score:  {result.get('overall_score')}/100")
            print(f"  Summary:        {result.get('summary')}")
            
            print("\n  Check Details:")
            for check in result.get("checks", []):
                match_icon = "[OK]" if check.get("match") else "[Mismatch]"
                print(f"    {match_icon} {check.get('field')}: {check.get('proposal_value')} vs {check.get('document_value')} ({check.get('score')} pts)")
                print(f"        Reasoning: {check.get('reasoning')}")
        else:
            print(f"\n[Error] Verification failed. Status code {response.status_code}: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[Error] Verification request threw exception: {e}")
        sys.exit(1)

    # 2. Test S3 Submission (if score is high enough or for testing purposes)
    print("\n[Test 2] Testing S3 storage submission...")
    proposal_id = proposal_data["proposal_no"]
    
    # Attach verification results mock or actual
    proposal_data["document_verification_result"] = result
    
    try:
        s3_url = f"{BACKEND_URL}/api/proposals/{proposal_id}"
        s3_resp = requests.post(s3_url, json=proposal_data, timeout=10)
        
        if s3_resp.status_code == 200:
            print(f"[Result] S3 Save request completed successfully: {s3_resp.json()}")
        else:
            print(f"[Error] S3 Save failed. Status code {s3_resp.status_code}: {s3_resp.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[Error] S3 Save threw exception: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("    Integration Test Successful!")
    print("=" * 60)

if __name__ == "__main__":
    main()
