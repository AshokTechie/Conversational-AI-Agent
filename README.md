# Conversational AI Agent – Agentic RAG System

An enterprise-ready, production-grade **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **Qdrant Vector Database**, **HuggingFace Embeddings**, **Google Gemini API**, and **Edge-TTS Voice I/O**.

---

## 🌟 Key Architecture & Highlights

- **LangGraph Agentic State Machine**: Employs cyclic state routing with 5 decoupled nodes:
  - **Node 1: Ingestion**: High-fidelity multi-engine PDF parsing (`PyMuPDF` with fallback to `pdfplumber`).
  - **Node 2: Chunking & Embedding**: Recursive character splitting with page provenance + local HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) via FastEmbed ONNX Runtime.
  - **Node 3: Retrieval**: Cosine similarity search in Qdrant with HNSW indexing, guaranteeing **sub-200ms retrieval latency**.
  - **Node 4: Relevance Grader & Router**: Self-corrective reasoning that detects poor retrieval and routes queries to the **Query Rewriter** for iterative reformulation (up to 2 retries).
  - **Node 5: Generation & Output Validation**: Grounded answer synthesis with Google Gemini API, Pydantic schema validation, and citation enforcement (`[Page X]`).
- **Graceful Fallback**: Safely rejects out-of-scope or unanswerable queries without hallucinations.
- **Dual-Channel Voice I/O**:
  - **Speech-to-Text (STT)**: Whisper / browser microphone transcription.
  - **Text-to-Speech (TTS)**: Microsoft Edge Neural Voices (`en-US-AriaNeural`) with zero API key dependencies.
- **Automated Benchmark Suite**: 22 diverse test scenarios covering single-hop, multi-hop, ambiguous, out-of-scope, and adversarial prompts (targeting **95%+ accuracy**).
- **Observable Dashboard**: Streamlit interface displaying real-time node transitions, step traces, confidence metrics, and audio player.

---

## 📂 Project Directory Structure

```
Conversational-AI-Agent/
├── benchmark_docs/                 # Benchmark PDFs and test fixtures
├── data/                           # Qdrant local database, audio files, benchmark JSON
├── src/
│   ├── config.py                   # Central settings & hyperparameters
│   ├── ingestion/
│   │   └── pdf_loader.py           # Node 1: Multi-engine PDF ingestion
│   ├── chunking_embedding/
│   │   ├── chunker.py              # Node 2A: Recursive character chunker
│   │   └── embedder.py             # Node 2B: HuggingFace / FastEmbed ONNX embedder
│   ├── vectorstore/
│   │   └── qdrant_store.py         # Node 3: Qdrant HNSW vector store (<200ms)
│   ├── agent/
│   │   ├── state.py                # LangGraph AgentState TypedDict
│   │   ├── prompts.py              # Grader, Rewriter, Synthesis prompts
│   │   ├── validation.py           # Pydantic schemas (GroundedAnswerSchema)
│   │   ├── nodes.py                # Decoupled agentic nodes & fallback
│   │   ├── conditional_edges.py    # Routing logic (grade check, retry limits)
│   │   └── graph.py                # Compiled LangGraph workflow
│   ├── voice/
│   │   ├── stt.py                  # Whisper / Gemini multimodal speech-to-text
│   │   └── tts.py                  # Edge-TTS neural speech synthesis
│   ├── eval/
│   │   ├── test_dataset.py         # 22 Benchmark Scenarios
│   │   ├── metrics.py              # Hit-rate, latency, fallback scoring
│   │   └── benchmark_runner.py     # Automated evaluation runner
│   └── app.py                      # Interactive Streamlit application
├── tests/
│   └── test_nodes.py               # Decoupled unit tests & e2e test suite
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone and enter workspace
cd Conversational-AI-Agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_GEMINI_MODEL=gemini-1.5-flash
DEFAULT_TTS_VOICE=en-US-AriaNeural
```
*(Note: If `GEMINI_API_KEY` is not provided, the pipeline automatically falls back to deterministic local grounded synthesis for testing).*

### 3. Run Unit Tests
```bash
python -m pytest tests/test_nodes.py -v
```

### 4. Run Automated Evaluation Benchmark (22 Scenarios)
```bash
python -m src.eval.benchmark_runner
```

### 5. Launch Interactive Web UI
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.
