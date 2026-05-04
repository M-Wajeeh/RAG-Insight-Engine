import os

from dotenv import load_dotenv

load_dotenv()


# Paths
DATA_PATH = "data/raw"
CHROMA_DB_DIR = "chroma_db"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4
MMR_FETCH_K_MULTIPLIER = 2
DEFAULT_RETRIEVER = "similarity"

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0

# Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
