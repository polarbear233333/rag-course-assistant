import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, VECTOR_DB_PATH
from backend.document_loader import DocumentLoader
from backend.text_splitter import TextSplitter
from backend.vector_store import VectorStore


def main():
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        print("Create data/ and put PDF, PPTX, DOCX or TXT files inside it.")
        return

    loader = DocumentLoader(data_dir=str(data_dir))
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    vector_store = VectorStore(db_path=VECTOR_DB_PATH)
    vector_store.clear_collection()

    documents = loader.load_all_documents()
    if not documents:
        print("No documents found.")
        return

    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages/slides/docs and produced {len(chunks)} chunks.")
    vector_store.add_documents(chunks)
    print("Done. Start API with: uvicorn app:app --reload --port 8000")


if __name__ == "__main__":
    main()
