from typing import Optional
from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.nodes import AgenticRAGNodes
from src.agent.conditional_edges import route_after_grading, route_after_generation

def build_agentic_rag_graph(nodes: Optional[AgenticRAGNodes] = None):
    """
    Compiles the LangGraph Agentic RAG state machine with conditional routing,
    self-correction loops, and graceful fallback.
    """
    if nodes is None:
        nodes = AgenticRAGNodes()

    workflow = StateGraph(AgentState)

    # Register Decoupled Nodes
    workflow.add_node("retrieval", nodes.retrieval_node)
    workflow.add_node("grading", nodes.grading_reasoning_node)
    workflow.add_node("rewrite_query", nodes.query_rewriter_node)
    workflow.add_node("generate", nodes.generation_node)
    workflow.add_node("fallback", nodes.fallback_node)
    workflow.add_node("synthesize_voice", nodes.voice_synthesis_node)

    # Define Graph Edges & Flow
    workflow.add_edge(START, "retrieval")
    workflow.add_edge("retrieval", "grading")

    # Conditional Edge: After Grading Chunks
    workflow.add_conditional_edges(
        "grading",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback"
        }
    )

    # Loop back to retrieval after query rewriting
    workflow.add_edge("rewrite_query", "retrieval")

    # Conditional Edge: After Generation (Checking Groundedness)
    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "synthesize_voice": "synthesize_voice",
            "fallback": "fallback"
        }
    )

    # Both pathways conclude with voice synthesis
    workflow.add_edge("fallback", "synthesize_voice")
    workflow.add_edge("synthesize_voice", END)

    return workflow.compile()

class AgenticRAGPipeline:
    """
    Convenience wrapper to run queries through the compiled LangGraph pipeline.
    """

    def __init__(self, nodes: Optional[AgenticRAGNodes] = None):
        self.nodes = nodes or AgenticRAGNodes()
        self.graph = build_agentic_rag_graph(self.nodes)

    def run(self, question: str, max_retries: int = 2, skip_voice: bool = False) -> AgentState:
        """
        Executes end-to-end agentic workflow for a user query.
        """
        initial_state: AgentState = {
            "question": question,
            "current_query": question,
            "documents": [],
            "relevance_grade": "insufficient",
            "generation": "",
            "citations": [],
            "confidence_score": 0.0,
            "is_grounded": False,
            "retry_count": 0,
            "max_retries": max_retries,
            "fallback_triggered": False,
            "pipeline_trace": [],
            "audio_path": None,
            "skip_voice": skip_voice
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
