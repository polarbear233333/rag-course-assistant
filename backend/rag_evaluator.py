from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    from .rag_agent import RAGAgent
except ImportError:
    from rag_agent import RAGAgent


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())


def lexical_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    pc, gc = {}, {}
    for x in p: pc[x] = pc.get(x, 0) + 1
    for x in g: gc[x] = gc.get(x, 0) + 1
    overlap = sum(min(pc.get(k, 0), gc.get(k, 0)) for k in gc)
    if overlap == 0: return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return round(2 * precision * recall / (precision + recall), 4)


def citation_score(answer: str) -> float:
    patterns = [r"资料片段\s*\d+", r"P\d+", r"chunk\s*\d+", r"\.pdf", r"\.pptx"]
    hits = sum(1 for p in patterns if re.search(p, answer, re.I))
    return round(min(hits / 3, 1.0), 4)


def context_precision(answer: str, sources: List[Dict]) -> float:
    if not sources:
        return 0.0
    ans = set(_tokens(answer))
    vals = []
    for s in sources:
        src = set(_tokens(s.get("preview", "")))
        vals.append(len(ans & src) / max(len(src), 1))
    return round(sum(vals) / len(vals), 4)


def answer_relevance(question: str, answer: str) -> float:
    q = set(_tokens(question))
    a = set(_tokens(answer))
    if not q or not a: return 0.0
    return round(len(q & a) / len(q), 4)


@dataclass
class EvalResult:
    question: str
    f1: float
    answer_relevance: float
    context_precision: float
    citation_score: float
    final_score: float
    answer: str
    sources: List[Dict]


class RAGEvaluator:
    """Custom RAG evaluation with optional RAGAS-compatible metric names.

    It does not require paid judge models, so it can run locally after the RAG
    index is built. The metrics are intentionally transparent and suitable for
    GitHub demos and course projects.
    """

    def __init__(self, agent: Optional[RAGAgent] = None):
        self.agent = agent or RAGAgent()

    def evaluate_one(self, question: str, reference_answer: str = "", top_k: int = 5) -> EvalResult:
        out = self.agent.answer_question_with_sources(question, top_k=top_k)
        ans, sources = out["answer"], out["sources"]
        f1 = lexical_f1(ans, reference_answer) if reference_answer else 0.0
        rel = answer_relevance(question, ans)
        cp = context_precision(ans, sources)
        cite = citation_score(ans)
        final = round(0.25 * f1 + 0.25 * rel + 0.25 * cp + 0.25 * cite, 4)
        return EvalResult(question, f1, rel, cp, cite, final, ans, sources)

    def evaluate_dataset(self, dataset: List[Dict], top_k: int = 5) -> Dict:
        rows = [asdict(self.evaluate_one(x["question"], x.get("reference_answer", ""), top_k)) for x in dataset]
        avg = {}
        for k in ["f1", "answer_relevance", "context_precision", "citation_score", "final_score"]:
            avg[k] = round(sum(r[k] for r in rows) / max(len(rows), 1), 4)
        return {"summary": avg, "num_examples": len(rows), "results": rows}
