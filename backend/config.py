import os

# LLM / embedding configuration. Values can be overridden by environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-max")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-v3")

# Data and vector database paths.
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")

# Chunking configuration.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Retrieval configuration.
TOP_K = int(os.getenv("TOP_K", "5"))
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
RRF_K = int(os.getenv("RRF_K", "60"))
