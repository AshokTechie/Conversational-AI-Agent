from typing import Literal
from src.agent.state import AgentState
from src.config import MAX_RETRIES

def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "fallback"]:
    """
    Evaluates whether to proceed directly to answer generation,
    trigger a query rewriter retry, or fallback gracefully.
    """
    grade = state.get("relevance_grade", "irrelevant")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)

    if grade == "relevant":
        return "generate"

    # Grade is irrelevant or insufficient
    if retry_count < max_retries:
        return "rewrite_query"
    else:
        return "fallback"

def route_after_generation(state: AgentState) -> Literal["synthesize_voice", "fallback"]:
    """
    Validates groundedness of the generated answer before voice synthesis.
    """
    is_grounded = state.get("is_grounded", True)
    if is_grounded:
        return "synthesize_voice"
    return "fallback"
