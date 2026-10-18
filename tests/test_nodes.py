import pytest
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import PDFIngestionNode
from src.chunking_embedding.chunker import DocumentChunkerNode
from src.chunking_embedding.embedder import HuggingFaceEmbedderNode
from src.vectorstore.qdrant_store import QdrantVectorStoreNode
from src.agent.nodes import AgenticRAGNodes
from src.agent.graph import AgenticRAGPipeline
from src.voice.tts import VoiceSynthesisNode
from benchmark_docs.generate_benchmark_pdf import generate_pdf

@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "test_doc.pdf"
    generate_pdf(fn)
    return str(fn)

def test_node_1_ingestion(sample_pdf_path):
    loader = PDFIngestionNode()
    pages = loader.load_pdf(sample_pdf_path)
    assert len(pages) >= 3
    assert pages[0]["page_number"] == 1
    assert "Agentic RAG" in pages[0]["text"]
    assert "doc_id" in pages[0]

def test_node_2_chunker(sample_pdf_path):
    loader = PDFIngestionNode()
    pages = loader.load_pdf(sample_pdf_path)
    chunker = DocumentChunkerNode(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_documents(pages)
    assert len(chunks) > len(pages)
    assert all("chunk_id" in c for c in chunks)
    assert all("page_number" in c for c in chunks)

def test_node_3_qdrant_sub_200ms_latency():
    embedder = HuggingFaceEmbedderNode()
    vectorstore = QdrantVectorStoreNode(in_memory=True)
    
    sample_chunks = [
        {"chunk_id": f"c_{i}", "doc_id": "d1", "page_number": (i % 3) + 1, "text": f"Technical chunk description {i} about vector database Qdrant and fast search.", "source": "test"}
        for i in range(20)
    ]
    texts = [c["text"] for c in sample_chunks]
    embeddings = embedder.embed_documents(texts)
    indexed = vectorstore.index_chunks(sample_chunks, embeddings)
    assert indexed == 20

    # Test retrieval latency
    query_vector, embed_ms = embedder.embed_query("fast vector database Qdrant")
    results, search_ms = vectorstore.search(query_vector, top_k=4)
    
    total_retrieval_ms = embed_ms + search_ms
    print(f"\n[Retrieval Latency]: Embed={embed_ms:.2f}ms, Search={search_ms:.2f}ms, Total={total_retrieval_ms:.2f}ms")
    
    assert len(results) > 0
    assert total_retrieval_ms < 200.0, f"Expected sub-200ms latency, got {total_retrieval_ms}ms"

def test_node_4_grader_routing():
    nodes = AgenticRAGNodes()
    state = {
        "question": "What is Qdrant?",
        "current_query": "What is Qdrant?",
        "documents": [{"similarity_score": 0.85, "text": "Qdrant is a vector database.", "page_number": 2}],
        "relevance_grade": "insufficient",
        "retry_count": 0,
        "max_retries": 2,
        "pipeline_trace": []
    }
    updated = nodes.grading_reasoning_node(state)
    assert updated["relevance_grade"] == "relevant"

def test_node_5_generation_groundedness():
    nodes = AgenticRAGNodes()
    state = {
        "question": "What is Qdrant?",
        "current_query": "What is Qdrant?",
        "documents": [{"similarity_score": 0.90, "text": "Qdrant is a vector database.", "page_number": 2}],
        "pipeline_trace": []
    }
    updated = nodes.generation_node(state)
    assert updated["generation"] != ""
    assert 2 in updated["citations"]
    assert updated["is_grounded"] is True

def test_voice_tts():
    tts = VoiceSynthesisNode()
    cleaned = tts.clean_text_for_speech("This is **bold** text [Page 3] with `code`.")
    assert "bold" in cleaned
    assert "[Page 3]" not in cleaned
    assert "**" not in cleaned

def test_langgraph_pipeline_e2e(sample_pdf_path):
    loader = PDFIngestionNode()
    chunker = DocumentChunkerNode()
    embedder = HuggingFaceEmbedderNode()
    vectorstore = QdrantVectorStoreNode(in_memory=True)

    pages = loader.load_pdf(sample_pdf_path)
    chunks = chunker.chunk_documents(pages)
    embeddings = embedder.embed_documents([c["text"] for c in chunks])
    vectorstore.index_chunks(chunks, embeddings)

    nodes = AgenticRAGNodes(embedder=embedder, vectorstore=vectorstore)
    pipeline = AgenticRAGPipeline(nodes)

    # In-scope question
    result = pipeline.run("Which vector database is deployed for similarity search?")
    assert "qdrant" in result["generation"].lower()
    assert result["fallback_triggered"] is False
    assert len(result["pipeline_trace"]) >= 3
