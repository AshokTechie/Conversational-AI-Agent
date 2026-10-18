from typing import List, Literal
from pydantic import BaseModel, Field

class RelevanceGradeSchema(BaseModel):
    """Schema for evaluating document relevance to user query."""
    grade: Literal["relevant", "irrelevant", "insufficient"] = Field(
        description="Whether the retrieved chunks contain enough factual context to answer the question."
    )
    reasoning: str = Field(
        description="Brief justification for the grade."
    )
    relevant_chunk_indices: List[int] = Field(
        default_factory=list,
        description="Indices (0-based) of the chunks that are actually relevant."
    )

class QueryRewriteSchema(BaseModel):
    """Schema for reformulating queries when initial retrieval is poor."""
    improved_query: str = Field(
        description="A more specific, clear, and keyword-rich version of the original query."
    )
    search_strategy: str = Field(
        description="Explanation of how the query was rewritten to target vector search."
    )

class GroundedAnswerSchema(BaseModel):
    """Schema for final synthesized answer with citation enforcement and groundedness."""
    answer: str = Field(
        description="The detailed, factual answer strictly derived from the provided context."
    )
    citations: List[int] = Field(
        default_factory=list,
        description="List of document page numbers directly cited in the answer."
    )
    confidence_score: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence that the answer is accurate and completely supported by context."
    )
    is_grounded: bool = Field(
        default=True,
        description="True if all claims are backed by the context, False if speculative."
    )
    direct_quotes: List[str] = Field(
        default_factory=list,
        description="Verbatim text snippets from the context backing up the core claims."
    )

class HallucinationGradeSchema(BaseModel):
    """Schema for self-reflection hallucination check."""
    is_grounded: bool = Field(
        description="True if the answer does NOT make up facts outside the retrieved context."
    )
    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="Any claims found in the answer that have no support in the context."
    )
