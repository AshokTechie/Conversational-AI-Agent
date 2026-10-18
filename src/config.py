import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DOCS_DIR = BASE_DIR / "benchmark_docs"
QDRANT_PATH = str(DATA_DIR / "qdrant_db")

# Gemini LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash")

# Embedding & Vector Database Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
QDRANT_COLLECTION_NAME = "agentic_pdf_rag"

# Chunking Parameters
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120

# Agentic Search & Reflection Limits
MAX_RETRIES = 2
DEFAULT_TOP_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.63  # Calibrated for BAAI/bge-small embedding cosine distribution

# Voice Synthesis (Edge-TTS)
DEFAULT_TTS_VOICE = "en-US-AriaNeural"  # High fidelity Microsoft Neural voice
AUDIO_OUTPUT_DIR = DATA_DIR / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
