"""
Generates the standard 3-page technical specification PDF used for automated benchmark testing.
"""

from pathlib import Path

PAGE_1_CONTENT = """Agentic RAG System Architecture & Technical Specification

1. System Overview & Objective
The Agentic RAG System is a goal-driven Conversational AI Agent built to process complex PDF documents, reason over contextual evidence, and synthesize grounded answers. Unlike traditional RAG pipelines that execute simple linear retrieval, this system uses an agentic architecture with conditional routing, self-correction, query rewriting, and graceful recovery when retrieval results are suboptimal.

2. LangGraph State Machine & Decoupled Nodes
The core orchestration is governed by a LangGraph state machine. The pipeline is architected around five independently testable and replaceable decoupled nodes:
- Node 1: Ingestion Node (extracts text and metadata from PDF files using PyMuPDF).
- Node 2: Chunking & Embedding Node (splits documents into recursive overlapping chunks and computes dense embeddings).
- Node 3: Retrieval Node (executes semantic similarity search on the vector store).
- Node 4: Relevance Grader & Router Node (evaluates chunk relevance and initiates query reformulation if retrieval is poor).
- Node 5: Generation & Output Validation Node (synthesizes grounded responses with strict Pydantic output validation).

3. State Management & Decoupled Architecture
State transitions are tracked across nodes with typed state structures storing original queries, retrieved chunks, reflection grades, retry counters, citations, and execution traces.
"""

PAGE_2_CONTENT = """Dense Vector Storage & HuggingFace Embedding Pipeline

1. Vector Store Selection: Qdrant
For semantic similarity search, the architecture utilizes Qdrant, a high-performance vector search engine implemented in Rust. Qdrant is configured with an HNSW (Hierarchical Navigable Small World) index with cosine distance metrics. This setup guarantees sub-200ms retrieval latency across 50+ indexed documents containing thousands of embedded chunks.

2. Local HuggingFace Embedding Generation
Embedding vectorization is performed locally using the HuggingFace BAAI/bge-small-en-v1.5 model (and sentence-transformers/all-MiniLM-L6-v2 fallback), projecting text chunks into 384-dimensional dense vector representations.
- Chunk Size: 600 tokens with 120 token overlap.
- Inference Speed: Running with FastEmbed / ONNX runtime achieves single-query embedding inference in approximately 12ms.
- Vector Lookup Time: Qdrant HNSW cosine lookup executes in 3ms to 5ms on CPU.

3. Self-Correction & Query Rewriting Flow
When the Relevance Grader grades retrieved chunks as 'irrelevant' or 'insufficient', the system does not fail silently. Instead, the Query Rewriter analyzes the original query, strips filler tokens, expands domain synonyms, and executes a second retrieval step (up to 2 retries). If relevant data remains absent after max retries, the system triggers graceful fallback.
"""

PAGE_3_CONTENT = """Generation Guardrails, Voice I/O & Benchmark Validation

1. Google Gemini API & Grounded Answer Synthesis
The system integrates Google Gemini API (gemini-1.5-flash / gemini-2.0-flash) for contextual reasoning and answer generation. The generation prompt enforces strict groundedness guardrails: answers must only cite facts explicitly present in the retrieved chunks. Pydantic schemas validate that page citations, direct quotes, and confidence scores are present in every response.

2. Voice I/O Workflow: Speech Recognition & Neural Synthesis
To support hands-free conversational interaction, the system provides a dual-channel Voice I/O workflow:
- Speech Recognition (STT): Powered by Whisper / Web Speech API, transcribing speech into structured text queries.
- Speech Synthesis (TTS): Powered by Edge-TTS (Microsoft Edge Neural Voices, such as en-US-AriaNeural), streaming realistic natural human audio responses with zero API key dependencies.

3. Automated Evaluation Benchmark (20+ Test Scenarios)
The system undergoes automated quality regression testing across 22 benchmark scenarios covering single-hop retrieval, multi-hop reasoning, ambiguous queries, and negative out-of-scope tests. The system achieves a 95%+ pass rate while strictly rejecting out-of-scope queries without hallucinations.
"""

def generate_pdf(output_path: Path):
    try:
        import fitz
        doc = fitz.open()
        
        pages = [PAGE_1_CONTENT, PAGE_2_CONTENT, PAGE_3_CONTENT]
        for p_idx, content in enumerate(pages):
            page = doc.new_page(width=595, height=842) # A4
            # Insert title & text
            rect = fitz.Rect(50, 50, 545, 792)
            page.insert_textbox(rect, content, fontsize=11, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)
            # Add footer with page number
            footer_rect = fitz.Rect(50, 800, 545, 820)
            page.insert_textbox(footer_rect, f"Page {p_idx + 1} of 3", fontsize=9, fontname="helv", align=fitz.TEXT_ALIGN_CENTER)
            
        doc.save(str(output_path))
        doc.close()
        print(f"Generated PDF successfully at {output_path}")
        return True
    except Exception as e:
        print(f"PyMuPDF generation failed ({e}), writing raw text fallback.")
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"--- PAGE 1 ---\n{PAGE_1_CONTENT}\n\n--- PAGE 2 ---\n{PAGE_2_CONTENT}\n\n--- PAGE 3 ---\n{PAGE_3_CONTENT}")
        return False

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "Agentic_RAG_System_Specification.pdf"
    generate_pdf(out_pdf)
