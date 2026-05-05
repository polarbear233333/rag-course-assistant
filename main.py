import os

from backend.config import MODEL_NAME, VECTOR_DB_PATH
from backend.rag_agent import RAGAgent


def main():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Vector DB not found. Run: python scripts/process_data.py")
        return
    agent = RAGAgent(model=MODEL_NAME)
    if agent.vector_store.get_collection_count() == 0:
        print("Vector DB is empty. Put course files under data/ and run: python scripts/process_data.py")
        return
    agent.chat()


if __name__ == "__main__":
    main()
