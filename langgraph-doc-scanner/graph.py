"""
graph.py — LangGraph graph definition.

Graph flow:
  preprocessor → paddle_ocr → classifier → vlm_extractor → validator → END

Conditional edge after classifier:
  - If doc_type is "unknown" AND OCR text is very short → go to END early
    (likely a blank or unreadable image, skip expensive VLM call)
  - Otherwise → proceed to vlm_extractor
"""

import logging
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes.preprocessor   import run_preprocessor
from nodes.paddle_ocr     import run_paddle_ocr
from nodes.classifier     import run_classifier
from nodes.vlm_extractor  import run_vlm_extractor
from nodes.validator      import run_validator

logger = logging.getLogger(__name__)

MIN_OCR_CHARS = 20  # Below this → skip VLM, not worth it


def _route_after_classifier(state: AgentState) -> str:
    """
    Conditional routing after classification.
    Skip VLM if we clearly can't read the document.
    """
    if state.get("error"):
        return "end"

    ocr_text = state.get("ocr_raw_text", "")
    doc_type = state.get("doc_type", "unknown")

    # Previous logic that skipped the VLM when OCR failed to detect text:
    # if doc_type == "unknown" and len(ocr_text) < MIN_OCR_CHARS:
    #     logger.warning("[Graph] OCR text too short and type unknown — skipping VLM")
    #     # Populate minimal output so validator still runs
    #     state["extracted_fields"] = {}
    #     return "validator"   # Go straight to validator to build error output

    return "vlm_extractor"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("preprocessor",   run_preprocessor)
    graph.add_node("paddle_ocr",     run_paddle_ocr)
    graph.add_node("classifier",     run_classifier)
    graph.add_node("vlm_extractor",  run_vlm_extractor)
    graph.add_node("validator",      run_validator)

    # Linear edges
    graph.set_entry_point("preprocessor")
    graph.add_edge("preprocessor", "paddle_ocr")
    graph.add_edge("paddle_ocr",   "classifier")

    # Conditional edge: classifier → vlm_extractor OR validator
    graph.add_conditional_edges(
        "classifier",
        _route_after_classifier,
        {
            "vlm_extractor": "vlm_extractor",
            "validator":     "validator",
            "end":           END,
        },
    )

    graph.add_edge("vlm_extractor", "validator")
    graph.add_edge("validator",     END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph