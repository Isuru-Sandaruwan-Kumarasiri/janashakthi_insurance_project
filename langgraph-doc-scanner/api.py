"""
api.py — FastAPI REST API for the Document Processing Agent.

Accepts document image uploads (or base64-encoded images) and returns
structured JSON output from the LangGraph OCR pipeline.

Endpoints:
    POST /process         — Upload an image file (multipart/form-data)
    POST /process/base64  — Submit a base64-encoded image (JSON body)
    GET  /health          — Health check for load balancers

Usage (local):
    uvicorn OCR.api:app --host 0.0.0.0 --port 8000

Usage (Docker):
    docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-... docagent-ocr
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure OCR package is importable
ocr_dir = Path(__file__).parent.absolute()
if str(ocr_dir) not in sys.path:
    sys.path.insert(0, str(ocr_dir))

from ocr_agent.document_agent import process_document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocAgent OCR API",
    description=(
        "Intelligent document processing API. Upload a scanned document "
        "(ID card, water bill, or medical report) and receive structured "
        "JSON with extracted data, validation results, and pipeline metadata."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for now; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "docagent_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Base64Request(BaseModel):
    """Request body for /process/base64 endpoint."""
    image: str  # base64-encoded image data
    filename: str = "upload.jpg"  # original filename (for extension detection)
    page: int = 0
    max_retries: int = 2
    model: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    timestamp: str
    version: str
    paddle_ocr: bool
    trocr: bool


# ---------------------------------------------------------------------------
# Startup event — warm up models
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Log startup info and verify critical dependencies."""
    logger.info("=" * 60)
    logger.info("DocAgent OCR API starting up...")
    logger.info(f"  Upload dir: {UPLOAD_DIR}")
    logger.info(f"  OPENROUTER_API_KEY set: {bool(os.getenv('OPENROUTER_API_KEY'))}")

    # Check PaddleOCR availability
    try:
        from medical_ocr import PADDLE_AVAILABLE, TROCR_AVAILABLE
        logger.info(f"  PaddleOCR available: {PADDLE_AVAILABLE}")
        logger.info(f"  TrOCR available:     {TROCR_AVAILABLE}")
    except ImportError as e:
        logger.warning(f"  Could not import medical_ocr: {e}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _save_upload(content: bytes, suffix: str) -> Path:
    """Save uploaded bytes to a temp file and return the path."""
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(content)
    return filepath


def _cleanup(filepath: Path) -> None:
    """Remove temp file after processing."""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception as e:
        logger.warning(f"Failed to clean up {filepath}: {e}")


def _run_pipeline(
    filepath: Path,
    page: int = 0,
    max_retries: int = 2,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the OCR pipeline and return the result dict."""
    # Override model if specified
    if model:
        os.environ["OPENROUTER_MODEL"] = model

    result = process_document(
        input_path=str(filepath),
        page_number=page,
        max_retries=max_retries,
    )
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    paddle_ok = False
    trocr_ok = False
    try:
        from medical_ocr import PADDLE_AVAILABLE, TROCR_AVAILABLE
        paddle_ok = PADDLE_AVAILABLE
        trocr_ok = TROCR_AVAILABLE
    except ImportError:
        pass

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        paddle_ocr=paddle_ok,
        trocr=trocr_ok,
    )


@app.post("/process", tags=["OCR Pipeline"])
async def process_image(
    file: UploadFile = File(..., description="Document image or PDF to process"),
    page: int = Query(0, ge=0, description="Page number for PDFs (0-indexed)"),
    max_retries: int = Query(2, ge=0, le=5, description="Max LLM extraction retries"),
    model: Optional[str] = Query(None, description="Override LLM model (e.g. openai/gpt-4o-mini)"),
):
    """
    Process a document image through the full OCR pipeline.

    Upload an image (PNG, JPG, TIFF, BMP, WEBP) or PDF file.
    Returns structured JSON with extracted data, validation results,
    and pipeline metadata.
    """
    # Validate file extension
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Save uploaded file
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    filepath = _save_upload(content, suffix)
    logger.info(f"Processing upload: {file.filename} ({len(content)} bytes) → {filepath}")

    try:
        result = _run_pipeline(filepath, page=page, max_retries=max_retries, model=model)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")
    finally:
        _cleanup(filepath)


@app.post("/process/base64", tags=["OCR Pipeline"])
async def process_base64(request: Base64Request):
    """
    Process a base64-encoded document image.

    Useful for API Gateway / Lambda integrations where multipart
    uploads are not convenient.
    """
    # Decode base64
    try:
        image_bytes = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data.")

    # Determine extension from filename
    suffix = Path(request.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".jpg"  # Default fallback

    filepath = _save_upload(image_bytes, suffix)
    logger.info(f"Processing base64 upload: {request.filename} ({len(image_bytes)} bytes)")

    try:
        result = _run_pipeline(
            filepath,
            page=request.page,
            max_retries=request.max_retries,
            model=request.model,
        )
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")
    finally:
        _cleanup(filepath)


# ---------------------------------------------------------------------------
# Main (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
