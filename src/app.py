import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path so 'src' module can be found
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

from src.config import (
    GEMINI_API_KEY, DEFAULT_GEMINI_MODEL, EMBEDDING_MODEL_NAME,
    DEFAULT_TTS_VOICE, AUDIO_OUTPUT_DIR
)
from src.ingestion.pdf_loader import PDFIngestionNode
from src.chunking_embedding.chunker import DocumentChunkerNode
from src.chunking_embedding.embedder import HuggingFaceEmbedderNode
from src.vectorstore.qdrant_store import QdrantVectorStoreNode
from src.agent.nodes import AgenticRAGNodes
from src.agent.graph import AgenticRAGPipeline
from src.voice.tts import VoiceSynthesisNode
from src.voice.stt import SpeechToTextNode
from src.eval.benchmark_runner import AutomatedBenchmarkRunner
from src.eval.test_dataset import BENCHMARK_SCENARIOS

# Page configuration
st.set_page_config(
    page_title="Conversational AI Agent – Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .badge-node {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        background-color: #0284c7;
        color: white;
    }
    .badge-success { background-color: #16a34a; }
    .badge-retry { background-color: #d97706; }
    .badge-fallback { background-color: #dc2626; }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_nodes" not in st.session_state:
    embedder = HuggingFaceEmbedderNode()
    vectorstore = QdrantVectorStoreNode()
    tts = VoiceSynthesisNode()
    nodes = AgenticRAGNodes(embedder=embedder, vectorstore=vectorstore, tts=tts)
    pipeline = AgenticRAGPipeline(nodes)
    st.session_state.pipeline_nodes = nodes
    st.session_state.pipeline = pipeline
if "indexed_docs_count" not in st.session_state:
    st.session_state.indexed_docs_count = st.session_state.pipeline_nodes.vectorstore.count()
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ System Control")
    st.markdown("### Agentic RAG Configuration")
    
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enter your Gemini API Key. If empty, the system runs with local deterministic fallback logic."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.session_state.pipeline_nodes.api_key = api_key_input

    st.markdown("---")
    st.markdown("### 📄 PDF Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        if st.button("Index PDF", use_container_width=True, disabled=(uploaded_file is None)):
            if uploaded_file:
                with st.spinner("Extracting & Vectorizing PDF..."):
                    # Save temp
                    temp_path = Path("data") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    loader = PDFIngestionNode()
                    chunker = DocumentChunkerNode()
                    pages = loader.load_pdf(str(temp_path))
                    chunks = chunker.chunk_documents(pages)
                    embeddings = st.session_state.pipeline_nodes.embedder.embed_documents([c["text"] for c in chunks])
                    count = st.session_state.pipeline_nodes.vectorstore.index_chunks(chunks, embeddings)
                    st.session_state.indexed_docs_count = st.session_state.pipeline_nodes.vectorstore.count()
                    st.success(f"Successfully indexed {count} chunks!")

    with col_idx2:
        if st.button("Index Benchmark Doc", use_container_width=True):
            with st.spinner("Indexing benchmark specification doc..."):
                bench_runner = AutomatedBenchmarkRunner()
                count = bench_runner.prepare_benchmark_documents()
                st.session_state.indexed_docs_count = st.session_state.pipeline_nodes.vectorstore.count()
                st.success(f"Indexed {count} benchmark chunks!")

    st.markdown(f"**Indexed Chunks in Qdrant:** `{st.session_state.indexed_docs_count}`")
    
    st.markdown("---")
    st.markdown("### 🎙️ Voice Synthesis Settings")
    enable_voice = st.checkbox("Enable Voice Responses (Edge-TTS)", value=True)
    voice_choice = st.selectbox(
        "Voice Profile",
        ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"],
        index=0
    )
    st.session_state.pipeline_nodes.tts.voice = voice_choice

# Main Header
st.title("🤖 Conversational AI Agent – Agentic RAG System")
st.caption("LangGraph State Machine • Qdrant Vector DB (<200ms) • HuggingFace Embeddings • Gemini API • Edge-TTS Voice I/O")

tab1, tab2, tab3 = st.tabs(["💬 Conversational Voice & Chat", "📊 20+ Automated Benchmark Evaluation", "🏛️ System Architecture"])

# ── Tab 1: Conversational Chat ──────────────────────────────────────────
with tab1:
    col_chat, col_trace = st.columns([1.6, 1.0])

    with col_chat:
        st.subheader("Interactive Conversation")
        
        # Audio Input Option
        audio_input = st.audio_input("Speak your question (Microphone)")
        voice_query = ""
        if audio_input is not None:
            with st.spinner("Transcribing audio input..."):
                stt_handler = SpeechToTextNode()
                temp_audio = Path("data/audio/temp_input.wav")
                temp_audio.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_audio, "wb") as f:
                    f.write(audio_input.getbuffer())
                voice_query = stt_handler.transcribe_audio_file(str(temp_audio))
                if voice_query:
                    st.info(f"🎙️ Transcribed Speech: '{voice_query}'")

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    st.caption(f"📚 Citations: Pages {msg['citations']}")
                if "audio_path" in msg and msg["audio_path"] and Path(msg["audio_path"]).exists():
                    st.audio(msg["audio_path"])

        # Input box
        user_query = st.chat_input("Ask a question about the indexed document...") or voice_query

        if user_query:
            # Append User Msg
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Execute LangGraph Pipeline
            with st.chat_message("assistant"):
                with st.spinner("Reasoning over document context..."):
                    start_t = time.perf_counter()
                    final_state = st.session_state.pipeline.run(user_query)
                    exec_ms = (time.perf_counter() - start_t) * 1000

                    generation = final_state.get("generation", "")
                    citations = final_state.get("citations", [])
                    audio_path = final_state.get("audio_path") if enable_voice else None
                    st.session_state.last_trace = final_state.get("pipeline_trace", [])

                    st.markdown(generation)
                    if citations:
                        st.caption(f"📚 Citations: Pages {citations} | Confidence: {final_state.get('confidence_score', 0.95)*100:.0f}%")

                    if audio_path and Path(audio_path).exists():
                        st.audio(audio_path)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": generation,
                        "citations": citations,
                        "audio_path": audio_path
                    })

    # Observable Pipeline Step-by-Step State Inspector
    with col_trace:
        st.subheader("🔍 Observable Pipeline Trace")
        st.caption("Live state machine inspection across decoupled nodes")

        if st.session_state.last_trace:
            for step in st.session_state.last_trace:
                node_name = step.get("node", "Node")
                status = step.get("status", "OK")
                badge_class = "badge-success" if "SUCCESS" in status or "SYNTHESIZED" in status else ("badge-retry" if "RETRY" in status else "badge-fallback")
                
                with st.expander(f"Step {step.get('step_id')}: {node_name} [{status}]", expanded=True):
                    st.markdown(f"<span class='badge-node {badge_class}'>{status}</span>", unsafe_allow_html=True)
                    st.json(step.get("details", {}))
        else:
            st.info("Submit a question to view real-time node transitions, latency breakdown, and routing decisions.")

# ── Tab 2: Automated Benchmark Evaluation ───────────────────────────────
with tab2:
    st.subheader("Automated Benchmark Suite: 20+ Test Scenarios")
    st.markdown("""
    This automated evaluation suite executes **22 comprehensive scenarios** covering:
    - **Single-Hop Factual Queries**: Verifies precision and recall on exact data points.
    - **Multi-Hop Cross-Page Reasoning**: Validates cross-page contextual synthesis.
    - **Negative / Out-of-Scope Tests**: Confirms zero-hallucination graceful fallback.
    - **Ambiguous Queries**: Validates Query Rewriter and Self-Correction loop.
    - **Adversarial Queries**: Checks prompt-injection and hallucination immunity.
    """)

    if st.button("🚀 Run 22-Scenario Automated Benchmark Suite", type="primary"):
        with st.spinner("Executing automated benchmark suite across all scenarios..."):
            runner = AutomatedBenchmarkRunner()
            summary = runner.run_benchmark()

            # KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{summary['accuracy_pct']}%", delta="Target: 95%+")
            c2.metric("Tests Passed", f"{summary['passed_scenarios']} / {summary['total_scenarios']}")
            c3.metric("Avg Retrieval Latency", f"{summary['avg_retrieval_latency_ms']} ms", delta="Target: <200ms")
            c4.metric("Sub-200ms Compliance", f"{summary['sub_200ms_compliance_pct']}%")

            # Table of Results
            st.markdown("### Detailed Scenario Breakdown")
            table_data = []
            for r in summary["detailed_results"]:
                table_data.append({
                    "Status": "✅ PASS" if r["passed"] else "❌ FAIL",
                    "Scenario ID": r["scenario_id"],
                    "Category": r["category"],
                    "Question": r["question"],
                    "Hit Rate": r["hit_rate"],
                    "Latency (ms)": r["retrieval_latency_ms"],
                    "Under 200ms": "⚡ YES" if r["latency_under_200ms"] else "⚠️ NO",
                    "Answer": r["generated_answer"][:120] + "..."
                })
            st.dataframe(table_data, use_container_width=True)

# ── Tab 3: System Architecture ──────────────────────────────────────────
with tab3:
    st.subheader("Agentic RAG Architecture & 5 Decoupled Nodes")
    st.markdown("""
    ### 5 Decoupled Pipeline Nodes
    1. **Node 1: Ingestion Node** (`src/ingestion/pdf_loader.py`): PyMuPDF multi-engine PDF parser.
    2. **Node 2: Chunking & Indexing Node** (`src/chunking_embedding/`): Recursive chunker + FastEmbed HuggingFace embeddings (`BAAI/bge-small-en-v1.5`).
    3. **Node 3: Retrieval Node** (`src/vectorstore/qdrant_store.py`): Qdrant HNSW cosine similarity search (<200ms).
    4. **Node 4: Relevance Grader & Router** (`src/agent/nodes.py`): Self-corrective relevance evaluator and query reformulation loop.
    5. **Node 5: Generation & Validation Node** (`src/agent/nodes.py`): Gemini grounded synthesis with Pydantic citation enforcement.
    """)
    st.info("Each node is completely decoupled, with independent unit tests and replaceable components.")
