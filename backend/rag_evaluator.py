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
    pred_tokens, gold_tokens = _tokens(pred), _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts, gold_counts = {}, {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in gold_counts)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def citation_score(answer: str) -> float:
    patterns = [r"资料片段\s*\d+", r"P\d+", r"chunk\s*\d+", r"\.pdf", r"\.pptx", r"\.docx", r"\.txt"]
    hits = sum(1 for pattern in patterns if re.search(pattern, answer, re.I))
    return round(min(hits / 3, 1.0), 4)


def context_precision(answer: str, sources: List[Dict]) -> float:
    if not sources:
        return 0.0
    answer_tokens = set(_tokens(answer))
    values = []
    for source in sources:
        source_tokens = set(_tokens(source.get("preview", "")))
        values.append(len(answer_tokens & source_tokens) / max(len(source_tokens), 1))
    return round(sum(values) / len(values), 4)


def answer_relevance(question: str, answer: str) -> float:
    question_tokens = set(_tokens(question))
    answer_tokens = set(_tokens(answer))
    if not question_tokens or not answer_tokens:
        return 0.0
    return round(len(question_tokens & answer_tokens) / len(question_tokens), 4)


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
    """Transparent custom RAG evaluation."""

    def __init__(self, agent: Optional[RAGAgent] = None):
        self.agent = agent or RAGAgent()

    def evaluate_one(self, question: str, reference_answer: str = "", top_k: int = 5) -> EvalResult:
        out = self.agent.answer_question_with_sources(question, top_k=top_k)
        answer, sources = out["answer"], out["sources"]
        f1 = lexical_f1(answer, reference_answer) if reference_answer else 0.0
        relevance = answer_relevance(question, answer)
        precision = context_precision(answer, sources)
        citation = citation_score(answer)
        final = round(0.25 * f1 + 0.25 * relevance + 0.25 * precision + 0.25 * citation, 4)
        return EvalResult(question, f1, relevance, precision, citation, final, answer, sources)

    def evaluate_dataset(self, dataset: List[Dict], top_k: int = 5) -> Dict:
        rows = [asdict(self.evaluate_one(item["question"], item.get("reference_answer", ""), top_k)) for item in dataset]
        avg = {}
        for key in ["f1", "answer_relevance", "context_precision", "citation_score", "final_score"]:
            avg[key] = round(sum(row[key] for row in rows) / max(len(rows), 1), 4)
        return {"summary": avg, "num_examples": len(rows), "results": rows}
