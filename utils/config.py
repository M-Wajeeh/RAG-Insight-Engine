import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# PATHS
# =========================
DATA_PATH = "data/raw"
CHROMA_DB_DIR = "chroma_db"

# =========================
# CHUNKING
# =========================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =========================
# RETRIEVAL
# =========================
TOP_K = 4

# =========================
# EMBEDDINGS
# =========================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# LLM
# =========================
GROQ_MODEL = "llama3-70b-8192"
TEMPERATURE = 0

# =========================
# API KEYS
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")