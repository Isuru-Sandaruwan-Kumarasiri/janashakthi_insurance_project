# import json
# import logging
# from pathlib import Path
# from fastapi import FastAPI, File, Form, HTTPException, Body, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Any

# import sys
# _ROOT_DIR = Path(__file__).resolve().parent.parent
# if str(_ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(_ROOT_DIR))

# from s3_storage import s3_manager
# from document_verification_agent import verify_documents

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Proposal Storage & Verification API",
#     description="Backend for saving proposals to AWS S3 and verifying uploaded documents via OCR + LLM.",
#     version="2.0.0"
# )

# # Allow requests from the Streamlit frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Restrict this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}


# # ─────────────────────────────────────────────
# # Health Check
# # ─────────────────────────────────────────────

# @app.get("/health")
# async def health_check():
#     """Verify the API is running and S3 is configured."""
#     return {
#         "status": "healthy",
#         "s3_bucket_configured": bool(s3_manager.bucket_name)
#     }


# # ─────────────────────────────────────────────
# # Proposal CRUD (S3)
# # ─────────────────────────────────────────────

# @app.post("/api/proposals/{proposal_id}")
# async def save_proposal(proposal_id: str, payload: Dict[str, Any] = Body(...)):
#     """
#     Saves the JSON proposal payload to AWS S3.
#     """
#     try:
#         success = s3_manager.save_proposal(proposal_id, payload)
#         if success:
#             return {"message": f"Proposal {proposal_id} successfully saved to S3.", "proposal_id": proposal_id}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to save proposal to S3.")
#     except ValueError as e:
#         logger.error(f"Configuration error: {e}")
#         raise HTTPException(status_code=500, detail="Server configuration error. S3 bucket not set.")
#     except Exception as e:
#         logger.error(f"S3 Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/api/proposals/{proposal_id}")
# async def get_proposal(proposal_id: str):
#     """
#     Fetches the JSON proposal from AWS S3 by its ID.
#     """
#     try:
#         data = s3_manager.get_proposal(proposal_id)
#         if data is None:
#             raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found.")
#         return data
#     except ValueError as e:
#         logger.error(f"Configuration error: {e}")
#         raise HTTPException(status_code=500, detail="Server configuration error. S3 bucket not set.")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"S3 Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# # ─────────────────────────────────────────────
# # Document Verification
# # ─────────────────────────────────────────────

# @app.post("/api/verify-documents")
# async def verify_documents_endpoint(
#     id_card_image: UploadFile = File(..., description="ID card image (jpg, png, pdf)"),
#     utility_bill_image: UploadFile = File(..., description="Utility bill image (jpg, png, pdf)"),
#     proposal_data: str = Form(..., description="Proposal form data as a JSON string"),
# ):
#     """
#     Verify uploaded ID card and utility bill against proposal form data.

#     1. Runs OCR on both uploaded images.
#     2. Uses an LLM to compare extracted data against the proposal form.
#     3. Returns a structured verification report.
#     """
#     # Validate file types
#     for upload, label in [(id_card_image, "ID card"), (utility_bill_image, "Utility bill")]:
#         suffix = Path(upload.filename or "file.jpg").suffix.lower()
#         if suffix not in SUPPORTED_EXTENSIONS:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported {label} file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
#             )

#     # Parse proposal data JSON
#     try:
#         proposal = json.loads(proposal_data)
#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=400, detail=f"Invalid proposal_data JSON: {e}")

#     # Read uploaded file bytes
#     id_card_bytes = await id_card_image.read()
#     utility_bill_bytes = await utility_bill_image.read()

#     if not id_card_bytes:
#         raise HTTPException(status_code=400, detail="Empty ID card image uploaded.")
#     if not utility_bill_bytes:
#         raise HTTPException(status_code=400, detail="Empty utility bill image uploaded.")

#     id_card_suffix = Path(id_card_image.filename or "file.jpg").suffix.lower()
#     utility_bill_suffix = Path(utility_bill_image.filename or "file.jpg").suffix.lower()

#     logger.info(f"Verification request: ID card ({len(id_card_bytes)} bytes), Utility bill ({len(utility_bill_bytes)} bytes)")

#     # Run the verification agent
#     try:
#         result = verify_documents(
#             proposal_data=proposal,
#             id_card_bytes=id_card_bytes,
#             id_card_suffix=id_card_suffix,
#             utility_bill_bytes=utility_bill_bytes,
#             utility_bill_suffix=utility_bill_suffix,
#         )
#         return result
#     except Exception as e:
#         logger.exception(f"Verification error: {e}")
#         raise HTTPException(status_code=500, detail=f"Document verification failed: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     import os
#     cwd = os.getcwd()
#     if os.path.basename(cwd) == "backend":
#         uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
#     else:
#         uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


import json
import logging
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Body, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import Dict, Any
import sys

# Ensure the root directory is in the path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from s3_storage import s3_manager
from document_verification_agent import verify_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Janashakthi Document Verification API",
    description="Backend for saving proposals to S3 and verifying documents via LangGraph OCR + LLM.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "s3_bucket_configured": bool(s3_manager.bucket_name)
    }

@app.post("/api/proposals/{proposal_id}")
async def save_proposal(proposal_id: str, payload: Dict[str, Any] = Body(...)):
    try:
        success = s3_manager.save_proposal(proposal_id, payload)
        if success:
            return {"message": f"Proposal {proposal_id} successfully saved to S3.", "proposal_id": proposal_id}
        raise HTTPException(status_code=500, detail="Failed to save proposal to S3.")
    except Exception as e:
        logger.error(f"S3 Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    try:
        data = s3_manager.get_proposal(proposal_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify-documents")
async def verify_documents_endpoint(
    id_card_image: UploadFile = File(..., description="ID card image"),
    utility_bill_image: UploadFile = File(..., description="Utility bill image"),
    proposal_data: str = Form(..., description="Proposal form data as JSON"),
):
    # Validate extensions
    for upload, label in [(id_card_image, "ID card"), (utility_bill_image, "Utility bill")]:
        suffix = Path(upload.filename or "file.jpg").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported {label} file type: {suffix}")

    try:
        proposal = json.loads(proposal_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Read bytes asynchronously
    id_card_bytes = await id_card_image.read()
    utility_bill_bytes = await utility_bill_image.read()

    if not id_card_bytes or not utility_bill_bytes:
        raise HTTPException(status_code=400, detail="One or more uploaded images are empty.")

    id_card_suffix = Path(id_card_image.filename or "file.jpg").suffix.lower()
    utility_bill_suffix = Path(utility_bill_image.filename or "file.jpg").suffix.lower()

    logger.info("Dispatching OCR tasks to background threadpool...")

    try:
        # Run the heavy OCR/LLM logic in a separate thread
        result = await run_in_threadpool(
            verify_documents,
            proposal_data=proposal,
            id_card_bytes=id_card_bytes,
            id_card_suffix=id_card_suffix,
            utility_bill_bytes=utility_bill_bytes,
            utility_bill_suffix=utility_bill_suffix,
        )
        return result
    except Exception as e:
        logger.exception(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Document verification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    cwd = os.getcwd()
    if os.path.basename(cwd) == "backend":
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
