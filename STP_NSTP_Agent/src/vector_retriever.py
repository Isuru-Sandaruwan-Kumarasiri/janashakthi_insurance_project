# =========================================================
# src/vector_retriever.py
# Vector retrieval for similar historical evidence
#
# Important:
# - No latest_vector_store_config.json is required.
# - The newest Chroma folder inside vector_store/ is auto-detected.
# - Retrieved documents are past-data evidence only.
# - Do not use vector retrieval to check company rules.
# =========================================================

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sqlite3

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.config import VECTOR_STORE_DIR, DEFAULT_EMBEDDING_MODEL
from src.utils import clean_text, json_safe


def detect_latest_chroma_dir() -> Path:
    """
    Pick latest Chroma folder from project/vector_store.
    A valid folder must contain chroma.sqlite3.
    """
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(f"Vector store folder not found: {VECTOR_STORE_DIR}")

    candidates = [
        p for p in VECTOR_STORE_DIR.iterdir()
        if p.is_dir() and (p / "chroma.sqlite3").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No Chroma folder found inside {VECTOR_STORE_DIR}. "
            "Expected something like vector_store/chroma_underwriting_..."
        )

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def detect_chroma_collection_name(chroma_dir: Path) -> str:
    """
    Detect collection name from chroma.sqlite3.
    If not possible, fallback to 'underwriting_rag_collection'.
    """
    sqlite_path = chroma_dir / "chroma.sqlite3"
    fallback = "underwriting_rag_collection"

    if not sqlite_path.exists():
        return fallback

    try:
        conn = sqlite3.connect(sqlite_path)
        row = conn.execute("SELECT name FROM collections LIMIT 1").fetchone()
        conn.close()

        if row and row[0]:
            return str(row[0])
    except Exception:
        pass

    return fallback


def load_latest_vector_store() -> Tuple[Chroma, Dict[str, Any]]:
    chroma_dir = detect_latest_chroma_dir()
    collection_name = detect_chroma_collection_name(chroma_dir)
    embedding_model = DEFAULT_EMBEDDING_MODEL

    embeddings = OllamaEmbeddings(model=embedding_model)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )

    config = {
        "chroma_dir": str(chroma_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "resolution_note": (
            "Vector store was auto-detected from project/vector_store. "
            "No config JSON was used."
        )
    }

    return vector_store, config


def format_docs(docs: List[Any], max_docs: int = 8, max_chars_per_doc: int = 900) -> List[Dict[str, Any]]:
    formatted = []

    for doc in docs:
        content = clean_text(doc.page_content)
        metadata = dict(doc.metadata or {})

        # Guardrail: remove obvious rule documents from vector evidence.
        text_joined = (content + " " + str(metadata)).lower()
        if "rule id:" in text_joined or metadata.get("doc_type") == "rule":
            continue

        formatted.append({
            "doc_no": len(formatted) + 1,
            "metadata": metadata,
            "content": content[:max_chars_per_doc]
        })

        if len(formatted) >= max_docs:
            break

    return json_safe(formatted)


def retrieve_vector_past_context(
    vector_query: str,
    k_initial: int = 20,
    k_final: int = 8
) -> Dict[str, Any]:
    """
    Retrieve vector evidence using an LLM-generated query.

    This searches the existing vector DB, but filters obvious rule docs
    because rules are already in the prompt.
    """
    vector_store, vector_config = load_latest_vector_store()

    query = clean_text(vector_query)
    if not query:
        query = "similar underwriting past cases underwriter remarks required documents medical evidence"

    raw_docs = vector_store.similarity_search(query, k=k_initial)
    candidate_docs = format_docs(raw_docs, max_docs=k_final, max_chars_per_doc=900)

    return json_safe({
        "vector_query": query,
        "vector_config": vector_config,
        "raw_docs_count": len(raw_docs),
        "final_docs_count": len(candidate_docs),
        "candidate_vector_docs": candidate_docs,
        "retrieval_note": (
            "Vector evidence is retrieved using an LLM-generated query. "
            "Obvious rule documents are filtered because company rules are already embedded in the prompt."
        )
    })
