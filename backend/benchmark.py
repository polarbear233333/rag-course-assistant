from __future__ import annotations

from typing import Dict, List, Optional
from openai import OpenAI

try:
    from .config import MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, MAX_TOKENS
    from .rag_agent import RAGAgent
    from .rag_evaluator import answer_relevance, citation_score, context_precision
except ImportError:
    from config import MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, MAX_TOKENS
    from rag_agent import RAGAgent
    from rag_evaluator import answer_relevance, citation_score, context_precision


class BenchmarkRunner:
    def __init__(self, agent: Optional[RAGAgent] = None):
        self.agent = agent or RAGAgent()
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    def pure_llm_answer(self, question: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful algorithms course teaching assistant. Do not fabricate citations."},
                    {"role": "user", "content": question},
                ],
                temperature=0.4,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"Pure LLM call failed: {e}"

    def compare_one(self, question: str, top_k: int = 5) -> Dict:
        rag = self.agent.answer_question_with_sources(question, top_k=top_k)
        llm_answer = self.pure_llm_answer(question)
        return {
            "question": question,
            "rag_answer": rag["answer"],
            "pure_llm_answer": llm_answer,
            "rag_sources": rag["sources"],
            "scores": {
                "rag_answer_relevance": answer_relevance(question, rag["answer"]),
                "pure_llm_answer_relevance": answer_relevance(question, llm_answer),
                "rag_context_precision": context_precision(rag["answer"], rag["sources"]),
                "rag_citation_score": citation_score(rag["answer"]),
                "pure_llm_citation_score": citation_score(llm_answer),
            },
        }

    def compare_dataset(self, questions: List[str], top_k: int = 5) -> Dict:
        rows = [self.compare_one(q, top_k=top_k) for q in questions]
        keys = rows[0]["scores"].keys() if rows else []
        avg = {k: round(sum(r["scores"][k] for r in rows) / max(len(rows), 1), 4) for k in keys}
        return {"summary": avg, "num_examples": len(rows), "results": rows}
