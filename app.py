from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.agents import MultiAgentTutor
from backend.benchmark import BenchmarkRunner
from backend.config import runtime_config
from backend.learning_profile import profile_summary, update_profile
from backend.rag_agent import RAGAgent
from backend.rag_evaluator import RAGEvaluator
from chat_store import append_log, load_logs, update_answer

logger = logging.getLogger("uvicorn.error")

agent = RAGAgent()
tutor = MultiAgentTutor(agent)
evaluator = RAGEvaluator(agent)
benchmark_runner = BenchmarkRunner(agent)

app = FastAPI(
    title="RAG Course Assistant",
    description="Source-grounded course assistant with hybrid retrieval, multi-agent tutoring, evaluation, and learning analytics.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    history: List[Dict] = Field(default_factory=list)
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    mode: str = "qa_agent"


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class EvalExample(BaseModel):
    question: str
    reference_answer: str = ""


class EvalRequest(BaseModel):
    examples: List[EvalExample]
    top_k: int = 5


class BenchmarkRequest(BaseModel):
    questions: List[str]
    top_k: int = 5


def _format_recent_logs(limit: int) -> str:
    logs = load_logs()[-limit:]
    if not logs:
        return "当前还没有问答记录。"
    return "\n\n".join(
        f"[{idx + 1}] 时间: {item.get('timestamp', '')}\n问题: {item.get('question', '')}\n回答: {item.get('answer', '')}"
        for idx, item in enumerate(logs)
    )


@app.get("/")
def index():
    return FileResponse(Path("frontend") / "index.html")


@app.get("/health")
def health():
    config = runtime_config()
    return {
        "status": "ok",
        "vector_docs": agent.vector_store.get_collection_count(),
        "provider": config["provider"],
        "api_base": config["api_base"],
        "model": config["model"],
        "embedding_model": config["embedding_model"],
        "has_api_key": config["has_api_key"],
    }


@app.get("/config")
def get_config():
    config = runtime_config()
    config["vector_docs"] = agent.vector_store.get_collection_count()
    return config


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = tutor.answer(req.query, chat_history=req.history, top_k=req.top_k)
    update_profile(req.query, result.answer, result.mode)
    return ChatResponse(answer=result.answer, mode=result.mode)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    return {"sources": agent.retrieve_sources(req.query, top_k=req.top_k)}


@app.post("/chat_with_sources")
def chat_with_sources(req: ChatRequest):
    result = tutor.answer(req.query, chat_history=req.history, top_k=req.top_k)
    update_profile(req.query, result.answer, result.mode)
    return {"answer": result.answer, "sources": result.sources, "mode": result.mode}


@app.post("/chat_stream")
def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks):
    record_id = str(uuid4())
    append_log(
        {
            "id": record_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": req.query,
            "answer": "",
        }
    )
    full_answer: List[str] = []

    def gen():
        for delta in tutor.answer_stream(req.query, chat_history=req.history, top_k=req.top_k):
            full_answer.append(delta)
            yield delta

    def save_log_and_profile():
        try:
            answer = "".join(full_answer)
            update_answer(record_id, answer)
            mode = tutor.route(req.query)
            update_profile(req.query, answer, mode)
            logger.info("chat log/profile updated: %s", record_id)
        except Exception as exc:
            logger.exception("chat log/profile update failed: %s", exc)

    background_tasks.add_task(save_log_and_profile)
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/agent")
def agent_route(req: ChatRequest):
    result = tutor.answer(req.query, chat_history=req.history, top_k=req.top_k)
    update_profile(req.query, result.answer, result.mode)
    return {"mode": result.mode, "answer": result.answer, "sources": result.sources}


@app.get("/profile")
def get_profile():
    return profile_summary()


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    dataset = [example.model_dump() for example in req.examples]
    return evaluator.evaluate_dataset(dataset, top_k=req.top_k)


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    return benchmark_runner.compare_dataset(req.questions, top_k=req.top_k)


@app.post("/analyze/recent")
def analyze_recent():
    context = _format_recent_logs(limit=20)
    query = "请基于上述记录生成学生最近的学习总结：近期学习重点、薄弱点、复习建议、下一步练习方向。"
    summary = agent.generate_response(query=query, context=context)
    return {"result": summary, "profile": profile_summary()}


@app.post("/analyze/mistakes")
def analyze_mistakes():
    context = _format_recent_logs(limit=30)
    query = "请基于上述记录生成错题与误区分析：概念混淆、错误模式、对应知识点、推荐复习题。"
    summary = agent.generate_response(query=query, context=context)
    return {"result": summary, "profile": profile_summary()}
