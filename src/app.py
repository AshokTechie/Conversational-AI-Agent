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
    page_title="ChatGPT – Conversational AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom ChatGPT Aesthetic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean ChatGPT Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f9f9f9;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Modern Pill Buttons */
    .stButton>button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    /* User Profile Badge */
    .chatgpt-user-badge {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 12px;
        background-color: #f1f5f9;
        margin-top: 24px;
        border: 1px solid #e2e8f0;
    }
    .user-avatar-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #0f766e;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Hero Heading */
    .hero-greeting {
        text-align: center;
        margin-top: 6vh;
        margin-bottom: 28px;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 600;
        color: #1e293b;
        letter-spacing: -0.02em;
    }
    
    /* Suggestion cards */
    .stCard {
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        padding: 12px 18px;
    }
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
if "recent_topics" not in st.session_state:
    st.session_state.recent_topics = [
        "C++ Standard Exploration",
        "Digital Electronics Clock Definition",
        "Modscan Issue Diagnosis",
        "GitHub Session Access Check",
        "LangGraph Agentic RAG Flow"
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Sidebar (ChatGPT Style) ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 **ChatGPT**")
    
    # "+ New chat" button
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")
    
    # Document Ingestion Collapsible
    with st.expander("📄 **Document & Vector DB**", expanded=(st.session_state.indexed_docs_count == 0)):
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
        max_pages = st.number_input(
            "Max Pages to Index (0 = Entire PDF)",
            min_value=0,
            max_value=2000,
            value=30,
            help="For large books/standards (500+ pages), 30-50 pages index in seconds."
        )
        
        c_up1, c_up2 = st.columns(2)
        with c_up1:
            if st.button("Index PDF", use_container_width=True, disabled=(uploaded_file is None)):
                if uploaded_file:
                    temp_path = Path("data") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    prog = st.progress(0)
                    st_txt = st.empty()
                    st_txt.info(f"Extracting '{uploaded_file.name}'...")

                    loader = PDFIngestionNode()
                    chunker = DocumentChunkerNode()
                    pages = loader.load_pdf(str(temp_path), max_pages=max_pages if max_pages > 0 else None)
                    
                    st_txt.info(f"Chunking {len(pages)} pages...")
                    prog.progress(20)
                    chunks = chunker.chunk_documents(pages)
                    total_chunks = len(chunks)

                    batch_size = 64
                    embeddings = []
                    for i in range(0, total_chunks, batch_size):
                        batch_texts = [c["text"] for c in chunks[i:i+batch_size]]
                        batch_emb = st.session_state.pipeline_nodes.embedder.embed_documents(batch_texts)
                        embeddings.extend(batch_emb)
                        done = min(i + len(batch_texts), total_chunks)
                        pct = 20 + int((done / total_chunks) * 70)
                        prog.progress(min(pct, 90))
                        st_txt.info(f"Embedding: {done} / {total_chunks} chunks ({pct}%)...")

                    prog.progress(95)
                    st_txt.info("Saving to Qdrant vector database...")
                    count = st.session_state.pipeline_nodes.vectorstore.index_chunks(chunks, embeddings)
                    st.session_state.indexed_docs_count = st.session_state.pipeline_nodes.vectorstore.count()
                    prog.progress(100)
                    st_txt.empty()
                    prog.empty()
                    st.success(f"Indexed {count} chunks from {len(pages)} pages!")

        with c_up2:
            if st.button("Index Benchmark", use_container_width=True):
                with st.spinner("Indexing benchmark doc..."):
                    bench_runner = AutomatedBenchmarkRunner()
                    count = bench_runner.prepare_benchmark_documents()
                    st.session_state.indexed_docs_count = st.session_state.pipeline_nodes.vectorstore.count()
                    st.success(f"Indexed {count} benchmark chunks!")

        st.caption(f"📚 **Indexed Chunks in Qdrant:** `{st.session_state.indexed_docs_count}`")

    # Settings Collapsible
    with st.expander("⚙️ **Settings & Voice**", expanded=False):
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="If empty, local grounded RAG logic is used."
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.session_state.pipeline_nodes.api_key = api_key_input

        voice_mode = st.toggle("Enable Live Voice Responses", value=False, key="voice_mode")
        voice_choice = st.selectbox(
            "Voice Profile",
            ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"],
            index=0
        )
        st.session_state.pipeline_nodes.tts.voice = voice_choice

    # Recents List
    st.markdown("<p style='font-size: 12px; font-weight: 600; color: #64748b; margin-top: 14px; margin-bottom: 6px;'>RECENTS</p>", unsafe_allow_html=True)
    for topic in st.session_state.recent_topics:
        if st.button(f"💬 {topic}", key=f"rec_{topic}", use_container_width=True):
            st.session_state.pending_query = f"Tell me about {topic}"
            st.rerun()

    # User Profile at Bottom (Matching ChatGPT screenshot)
    st.markdown("""
    <div class="chatgpt-user-badge">
        <div class="user-avatar-circle">AH</div>
        <div>
            <div style="font-weight: 600; font-size: 14px; color: #1e293b;">Ashok H</div>
            <div style="font-size: 12px; color: #64748b;">Go Plan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs ────────────────────────────────────────────────────────────
tab_chat, tab_bench, tab_arch = st.tabs(["💬 Chat", "📊 Evaluation Benchmark", "🏛️ Architecture"])

# ── TAB 1: CHATGPT INTERFACE ─────────────────────────────────────────────
with tab_chat:
    voice_query = ""
    user_input_query = None

    # Process any pending query triggered from chips or recents
    if st.session_state.pending_query:
        user_input_query = st.session_state.pending_query
        st.session_state.pending_query = None

    # Case A: LANDING PAGE (Empty chat - Exactly like user's screenshot!)
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
        <div class="hero-greeting">
            <div class="hero-title">How can I help, Ashok?</div>
        </div>
        """, unsafe_allow_html=True)

        # Centered Pill Input Form
        with st.form(key="chatgpt_hero_form", clear_on_submit=True):
            col_in, col_send = st.columns([5.5, 0.8])
            with col_in:
                hero_text = st.text_input(
                    "Ask ChatGPT",
                    placeholder="Ask anything about the indexed document or enter a prompt...",
                    label_visibility="collapsed"
                )
            with col_send:
                hero_send = st.form_submit_button("Ask ➔", use_container_width=True)
                if hero_send and hero_text.strip():
                    user_input_query = hero_text.strip()

        # Voice Input option on landing page
        with st.expander("🎙️ Speak your question (Live Voice Input)", expanded=False):
            audio_hero = st.audio_input("Microphone Input", key="hero_mic")
            if audio_hero is not None:
                audio_bytes = audio_hero.getvalue()
                if st.session_state.get("last_hero_audio") != hash(audio_bytes):
                    st.session_state.last_hero_audio = hash(audio_bytes)
                    with st.spinner("Transcribing your speech..."):
                        stt_h = SpeechToTextNode()
                        tmp_a = Path("data/audio/temp_hero.wav")
                        tmp_a.parent.mkdir(parents=True, exist_ok=True)
                        with open(tmp_a, "wb") as f:
                            f.write(audio_bytes)
                        voice_query = stt_h.transcribe_audio_file(str(tmp_a))
                        if voice_query:
                            user_input_query = voice_query

        # Quick Suggestion Chips (like ChatGPT screenshot)
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("📄 Ask about C++ Standard", use_container_width=True):
                user_input_query = "What is C++ according to the specification?"
        with col_c2:
            if st.button("✍️ Summarize LangGraph Nodes", use_container_width=True):
                user_input_query = "Which 5 decoupled nodes form the LangGraph pipeline?"
        with col_c3:
            if st.button("⚡ Qdrant Target Latency", use_container_width=True):
                user_input_query = "What is the target retrieval latency for Qdrant?"

    # Case B: ACTIVE CONVERSATION (Chat history present)
    else:
        # Top Action Bar with Clear button
        col_h1, col_h2 = st.columns([4, 1])
        with col_h1:
            st.markdown("### 💬 **Conversation**")
        with col_h2:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Scrollable Chat Container
        chat_box = st.container(height=480)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "citations" in msg and msg["citations"]:
                        st.caption(f"📚 Citations: Pages {msg['citations']}")
                    if msg.get("audio_path") and Path(msg["audio_path"]).exists():
                        st.audio(msg["audio_path"])

        # Top/Bottom Input Form for active chat
        with st.form(key="chatgpt_active_form", clear_on_submit=True):
            col_in2, col_send2 = st.columns([5.5, 0.8])
            with col_in2:
                active_text = st.text_input(
                    "Ask ChatGPT",
                    placeholder="Ask a follow-up question...",
                    label_visibility="collapsed"
                )
            with col_send2:
                active_send = st.form_submit_button("Send ➔", use_container_width=True)
                if active_send and active_text.strip():
                    user_input_query = active_text.strip()

        # Optional Microphone
        with st.expander("🎙️ Speak your question (Live Voice Input)", expanded=False):
            audio_active = st.audio_input("Microphone Input", key="active_mic")
            if audio_active is not None:
                audio_bytes = audio_active.getvalue()
                if st.session_state.get("last_active_audio") != hash(audio_bytes):
                    st.session_state.last_active_audio = hash(audio_bytes)
                    with st.spinner("Transcribing speech..."):
                        stt_h = SpeechToTextNode()
                        tmp_a = Path("data/audio/temp_active.wav")
                        tmp_a.parent.mkdir(parents=True, exist_ok=True)
                        with open(tmp_a, "wb") as f:
                            f.write(audio_bytes)
                        voice_query = stt_h.transcribe_audio_file(str(tmp_a))
                        if voice_query:
                            user_input_query = voice_query

    # ── Pipeline Execution for any Query ──────────────────────────────────
    if user_input_query:
        # Check if voice should be synthesized: ONLY if asked via mic or voice_mode toggle is ON
        should_speak = bool(voice_query) or st.session_state.get("voice_mode", False)

        st.session_state.chat_history.append({"role": "user", "content": user_input_query})

        with st.spinner("Thinking and synthesizing answer..."):
            start_t = time.perf_counter()
            final_state = st.session_state.pipeline.run(
                user_input_query,
                skip_voice=(not should_speak)
            )
            exec_ms = (time.perf_counter() - start_t) * 1000

            generation = final_state.get("generation", "")
            citations = final_state.get("citations", [])
            audio_path = final_state.get("audio_path") if should_speak else None

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": generation,
                "citations": citations,
                "audio_path": audio_path
            })
        st.rerun()

# ── TAB 2: AUTOMATED BENCHMARK EVALUATION ────────────────────────────────
with tab_bench:
    st.subheader("📊 Automated Benchmark Suite (22 Scenarios)")
    st.markdown("""
    Evaluates: Single-Hop Factual, Multi-Hop Reasoning, Graceful Fallback, Ambiguous reformulation, and Adversarial Immunity.
    """)

    if st.button("🚀 Run 22-Scenario Automated Benchmark Suite", type="primary"):
        with st.spinner("Executing evaluation scenarios..."):
            runner = AutomatedBenchmarkRunner()
            summary = runner.run_benchmark()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Accuracy", f"{summary['accuracy_pct']}%", delta="Target: 95%+")
            c2.metric("Tests Passed", f"{summary['passed_scenarios']} / {summary['total_scenarios']}")
            c3.metric("Avg Retrieval Latency", f"{summary['avg_retrieval_latency_ms']} ms", delta="Target: <200ms")
            c4.metric("Sub-200ms Compliance", f"{summary['sub_200ms_compliance_pct']}%")

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

# ── TAB 3: SYSTEM ARCHITECTURE ───────────────────────────────────────────
with tab_arch:
    st.subheader("🏛️ Agentic RAG Architecture & 5 Decoupled Nodes")
    st.markdown("""
    1. **Node 1: Ingestion Node** (`src/ingestion/pdf_loader.py`): PyMuPDF multi-engine PDF parser.
    2. **Node 2: Chunking & Indexing Node** (`src/chunking_embedding/`): Recursive chunker + FastEmbed embeddings.
    3. **Node 3: Retrieval Node** (`src/vectorstore/qdrant_store.py`): Qdrant HNSW cosine similarity search (<200ms).
    4. **Node 4: Relevance Grader & Router** (`src/agent/nodes.py`): Self-corrective relevance evaluator & query reformulation.
    5. **Node 5: Generation & Validation Node** (`src/agent/nodes.py`): Gemini grounded synthesis with citation enforcement.
    """)
