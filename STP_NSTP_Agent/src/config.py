# =========================================================
# src/config.py
# Config for LLM-first STP/NSTP Rule Checking + RAG Document Agent
#
# No vector config JSON is required.
# SQLite DB path and vector_store folder are resolved from project root.
# =========================================================

from pathlib import Path
import os
from dotenv import load_dotenv


def get_project_dir() -> Path:
    """
    Project root is the parent folder of src/.
    PROJECT_DIR in .env is optional, but the code does not depend on it.
    """
    env_project_dir = os.getenv("PROJECT_DIR")
    if env_project_dir:
        return Path(env_project_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


TEMP_PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = TEMP_PROJECT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

PROJECT_DIR = get_project_dir()
ENV_PATH = PROJECT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_OUTPUT_DIR = PROCESSED_DIR / "final_agent_outputs"
DATABASE_DIR = PROJECT_DIR / "database"
VECTOR_STORE_DIR = PROJECT_DIR / "vector_store"
EXAMPLES_DIR = PROJECT_DIR / "examples"
TESTS_DIR = PROJECT_DIR / "tests"

DB_PATH = DATABASE_DIR / "underwriting_system.db"
FINAL_OUTPUT_PATH = FINAL_OUTPUT_DIR / "llm_rule_rag_agent_output.json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "qwen/qwen3-32b"
)

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

VALID_DECISIONS = ["STP", "NSTP"]


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)


def get_runtime_status() -> dict:
    ensure_directories()

    chroma_dirs = []
    if VECTOR_STORE_DIR.exists():
        chroma_dirs = [
            str(p.name)
            for p in VECTOR_STORE_DIR.iterdir()
            if p.is_dir() and (p / "chroma.sqlite3").exists()
        ]

    return {
        "project_dir": str(PROJECT_DIR),
        "env_path": str(ENV_PATH),
        "env_exists": ENV_PATH.exists(),
        "database_path": str(DB_PATH),
        "database_exists": DB_PATH.exists(),
        "vector_store_dir": str(VECTOR_STORE_DIR),
        "vector_store_dir_exists": VECTOR_STORE_DIR.exists(),
        "detected_chroma_folders": chroma_dirs,
        "detected_chroma_count": len(chroma_dirs),
        "final_output_path": str(FINAL_OUTPUT_PATH),
        "openrouter_model": OPENROUTER_MODEL,
        "openrouter_api_key_loaded": bool(OPENROUTER_API_KEY),
        "openrouter_api_key_prefix": OPENROUTER_API_KEY[:8] if OPENROUTER_API_KEY else None,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


def validate_runtime_files(strict: bool = False) -> dict:
    ensure_directories()
    status = get_runtime_status()

    missing = []
    if not status["env_exists"]:
        missing.append(".env file")
    if not status["openrouter_api_key_loaded"]:
        missing.append("OPENROUTER_API_KEY")
    if not status["database_exists"]:
        missing.append("database/underwriting_system.db")
    if status["detected_chroma_count"] == 0:
        missing.append("at least one Chroma folder inside vector_store/")

    status["missing_items"] = missing
    status["ready"] = len(missing) == 0

    if strict and missing:
        raise RuntimeError("Missing required runtime items: " + ", ".join(missing))

    return status


def print_config() -> None:
    status = validate_runtime_files(strict=False)

    print("====================================")
    print("LLM Rule RAG STP/NSTP Agent Config")
    print("====================================")
    print("Project dir:", status["project_dir"])
    print("Env exists:", status["env_exists"])
    print("Database exists:", status["database_exists"])
    print("Vector store dir exists:", status["vector_store_dir_exists"])
    print("Detected Chroma folders:", status["detected_chroma_folders"])
    print("OpenRouter model:", status["openrouter_model"])
    print("OpenRouter key loaded:", status["openrouter_api_key_loaded"])
    print("Embedding model:", status["embedding_model"])
    print("Ready:", status["ready"])

    if status["missing_items"]:
        print("Missing items:")
        for item in status["missing_items"]:
            print("-", item)
