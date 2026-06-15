import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible provider configuration.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://aihubmix.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "alicloud-deepseek-v4-flash")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))
PROVIDER_NAME = os.getenv("PROVIDER_NAME", "AIHubMix")

# Data and vector database paths.
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "course_rag_collection")

# Chunking configuration.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Retrieval configuration.
TOP_K = int(os.getenv("TOP_K", "5"))
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
RRF_K = int(os.getenv("RRF_K", "60"))


def runtime_config() -> dict:
    return {
        "provider": PROVIDER_NAME,
        "api_base": OPENAI_API_BASE,
        "model": MODEL_NAME,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "has_api_key": bool(OPENAI_API_KEY),
        "data_dir": DATA_DIR,
        "vector_db_path": VECTOR_DB_PATH,
        "collection_name": COLLECTION_NAME,
        "top_k": TOP_K,
    }
