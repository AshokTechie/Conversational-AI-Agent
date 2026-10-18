from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class PipelineStep(TypedDict):
    step_id: int
    node: str
    status: str
    timestamp: str
    details: Dict[str, Any]

class AgentState(TypedDict):
    """
    Central state for the LangGraph Agentic RAG pipeline.
    Maintains question, documents, reasoning, routing decisions, and validation.
    """
    question: str
    current_query: str
    documents: List[Dict[str, Any]]
    relevance_grade: str          # "relevant" | "irrelevant" | "insufficient"
    generation: str
    citations: List[int]
    confidence_score: float
    is_grounded: bool
    retry_count: int
    max_retries: int
    fallback_triggered: bool
    pipeline_trace: List[PipelineStep]
    audio_path: Optional[str]
    skip_voice: Optional[bool]
