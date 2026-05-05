from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
from uuid import uuid4
import logging

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.agents import MultiAgentTutor
from backend.benchmark import BenchmarkRunner
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
    title="Hybrid RAG Course Tutor",
    description="Source-grounded RAG system with streaming, multi-agent tutoring, evaluation, learning profile and RAG-vs-LLM benchmark.",
    version="2.0.0",
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
    history: List[Dict] = []
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


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "vector_docs": agent.vector_store.get_collection_count()}


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
    append_log({"id": record_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "question": req.query, "answer": ""})
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
            logger.warning(f"chat_log/profile updated id={record_id}")
        except Exception as e:
            logger.exception(f"chat_log/profile update failed: {e}")

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
    dataset = [x.model_dump() for x in req.examples]
    return evaluator.evaluate_dataset(dataset, top_k=req.top_k)


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    return benchmark_runner.compare_dataset(req.questions, top_k=req.top_k)


@app.post("/analyze/recent")
def analyze_recent():
    logs = load_logs()[-20:]
    context = "当前还没有任何问答记录。" if not logs else "\n".join(
        f"[{i+1}] 时间：{x.get('timestamp','')}\n问题：{x.get('question','')}\n回答：{x.get('answer','')}" for i, x in enumerate(logs)
    )
    query = "请基于上述记录生成学生最近的学习总结：近期学习重点、薄弱点、复习建议、下一步练习方向。"
    summary = agent.generate_response(query=query, context=context)
    return {"result": summary, "profile": profile_summary()}


@app.post("/analyze/mistakes")
def analyze_mistakes():
    logs = load_logs()[-30:]
    context = "当前还没有任何问答记录。" if not logs else "\n".join(
        f"[{i+1}] 时间：{x.get('timestamp','')}\n问题：{x.get('question','')}\n回答：{x.get('answer','')}" for i, x in enumerate(logs)
    )
    query = "请基于上述记录生成错题/误区分析：概念混淆、错误模式、对应知识点、推荐复习题。"
    summary = agent.generate_response(query=query, context=context)
    return {"result": summary, "profile": profile_summary()}
