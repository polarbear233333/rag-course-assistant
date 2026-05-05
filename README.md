# Hybrid RAG Course Tutor

A production-style Retrieval-Augmented Generation system for course learning scenarios. The project upgrades a classroom NLP/RAG assignment into a GitHub- and resume-ready application with hybrid retrieval, source-grounded generation, streaming FastAPI responses, multi-agent tutoring, custom RAG evaluation, learning profile analytics, and RAG-vs-Pure-LLM benchmarking.

## Highlights

- **Hybrid retrieval**: combines dense embedding retrieval and BM25 keyword retrieval with Reciprocal Rank Fusion (RRF).
- **Source-grounded QA**: answers are generated from retrieved course snippets and are encouraged to cite filename, page number, and chunk id.
- **Streaming FastAPI backend**: `/chat_stream` streams model tokens to the browser, creating a ChatGPT-like user experience.
- **Multi-agent tutor**: lightweight router automatically switches between QA Agent, Exercise Generation Agent, Mistake Diagnosis Agent, and Learning Profile Agent.
- **RAG evaluation**: transparent custom metrics inspired by RAGAS-style dimensions, including answer relevance, context precision, citation score, lexical F1, and final score.
- **Learning profile**: tracks user questions, detects topic distribution, and summarizes review candidates.
- **Benchmark pipeline**: compares Pure LLM and source-grounded RAG on the same questions.
- **Document ingestion**: supports PDF, PPTX, DOCX, and TXT; PDF/PPT images can be OCR-enhanced with Tesseract.

## System Architecture

```text
course files /data
      |
      v
DocumentLoader -> Semantic-aware TextSplitter -> Chroma Vector DB
      |                                      |
      |                                      v
      |                         Dense Retrieval + BM25
      |                                      |
      v                                      v
FastAPI Backend <---- RRF Hybrid Retriever <---- Multi-Agent Tutor
      |
      +-- Streaming Chat UI
      +-- RAG Evaluation
      +-- Learning Profile
      +-- Pure LLM vs RAG Benchmark
```

## Project Structure

```text
.
├── app.py                         # FastAPI application and API endpoints
├── main.py                        # CLI chat entry point
├── chat_store.py                  # Local chat log storage
├── backend/
│   ├── rag_agent.py               # Source-grounded RAG generation
│   ├── vector_store.py            # Chroma + Dense/BM25/RRF retrieval
│   ├── document_loader.py         # PDF/PPTX/DOCX/TXT loader + OCR hooks
│   ├── text_splitter.py           # Semantic-aware chunking
│   ├── agents.py                  # Multi-agent routing and prompt wrappers
│   ├── rag_evaluator.py           # Custom RAG evaluation metrics
│   ├── benchmark.py               # Pure LLM vs RAG comparison
│   ├── learning_profile.py        # User learning profile analytics
│   └── config.py                  # Environment-driven configuration
├── scripts/
│   ├── process_data.py            # Build vector database
│   ├── evaluate.py                # Run RAG evaluation dataset
│   └── benchmark.py               # Run RAG-vs-LLM benchmark
├── frontend/
│   └── index.html                 # ChatGPT-like web UI
├── datasets/
│   └── eval_examples.jsonl        # Example evaluation dataset
├── data/                          # Put course documents here
├── outputs/                       # Evaluation / benchmark outputs
├── requirements.txt
└── .env.example
```

## Installation

```bash
conda create -n hybrid-rag python=3.10 -y
conda activate hybrid-rag
pip install -r requirements.txt
```

Create a `.env` file or set environment variables:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3-max
OPENAI_EMBEDDING_MODEL=text-embedding-v3
```

> The default configuration uses the DashScope OpenAI-compatible endpoint, but any OpenAI-compatible chat and embedding endpoint can be used.

## Build the Knowledge Base

Put course materials into `data/`:

```text
data/
├── LectureNotes.pdf
├── Slides01.pptx
└── ...
```

Then run:

```bash
python scripts/process_data.py
```

This loads documents, chunks them, generates embeddings, and writes them into the Chroma vector database under `vector_db/`.

## Run the Web App

```bash
uvicorn app:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Main API Endpoints

| Endpoint | Method | Description |
|---|---:|---|
| `/chat_stream` | POST | Streaming ChatGPT-like answer generation |
| `/chat_with_sources` | POST | Returns answer + retrieved source snippets |
| `/retrieve` | POST | Debug top-k hybrid retrieval results |
| `/agent` | POST | Multi-agent routed answer |
| `/profile` | GET | Learning profile and topic distribution |
| `/evaluate` | POST | Custom RAG evaluation over examples |
| `/benchmark` | POST | Compare Pure LLM vs RAG |
| `/analyze/recent` | POST | Generate recent learning summary |
| `/analyze/mistakes` | POST | Generate mistake review |

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat_with_sources \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain the Master Theorem", "top_k":5}'
```

## RAG Evaluation

Run the example dataset:

```bash
python scripts/evaluate.py \
  --dataset datasets/eval_examples.jsonl \
  --output outputs/eval_results.json
```

The evaluator reports:

- `answer_relevance`: overlap between question and generated answer.
- `context_precision`: whether the answer is supported by retrieved contexts.
- `citation_score`: whether answer includes traceable source evidence.
- `f1`: lexical overlap with a reference answer, when available.
- `final_score`: averaged composite score.

The dataset format is JSONL:

```json
{"question":"Explain Dijkstra's algorithm.","reference_answer":"Dijkstra solves single-source shortest paths with non-negative weights."}
```

## Benchmark: Pure LLM vs RAG

```bash
python scripts/benchmark.py \
  --dataset datasets/eval_examples.jsonl \
  --output outputs/benchmark_results.json
```

The benchmark runs both modes on the same questions:

1. **Pure LLM**: no retrieved context.
2. **Source-grounded RAG**: Dense + BM25 + RRF context is injected before generation.

This makes the project more than a demo: it contains a reproducible evaluation pipeline.

## Multi-Agent Design

The system uses an inspectable router instead of a heavy agent framework:

- **QA Agent**: explains course concepts with citations.
- **Exercise Agent**: generates practice problems, hints, and reference answers.
- **Mistake Agent**: diagnoses misconceptions and suggests targeted fixes.
- **Profile Agent**: summarizes learning focus and review strategy.

This design keeps the system easy to debug and easy to explain in interviews.

## Resume Description

**Hybrid RAG Course Tutor** — Built a production-style course assistant using Dense Retrieval + BM25 + Reciprocal Rank Fusion, enabling source-grounded QA, exercise generation, mistake diagnosis, learning-profile analytics, and RAG-vs-Pure-LLM benchmarking. Implemented a FastAPI streaming backend, Chroma vector database, multi-format document ingestion, custom RAG evaluation metrics, and a web interface for real-time interaction and citation inspection.

## Notes

- If OCR is needed, install Tesseract locally and make sure the language packs are available.
- If NLTK tokenization fails, run `python download_nltk_punkt.py` or let the fallback regex tokenizer handle English tokens.
- The project intentionally uses transparent custom evaluation metrics so that it can run without requiring an extra LLM judge.
