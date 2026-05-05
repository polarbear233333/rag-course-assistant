from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

try:
    from .rag_agent import RAGAgent
except ImportError:
    from rag_agent import RAGAgent


@dataclass
class AgentResult:
    mode: str
    answer: str
    sources: List[Dict]


class MultiAgentTutor:
    """Lightweight multi-agent orchestration for a course RAG tutor.

    The system is intentionally simple and inspectable: a router chooses a
    mode, then specialized prompt wrappers call the same source-grounded RAG
    backend. This is easier to explain in a resume/interview than a black-box
    agent framework.
    """

    def __init__(self, rag_agent: Optional[RAGAgent] = None):
        self.rag = rag_agent or RAGAgent()

    def route(self, query: str) -> str:
        q = query.lower()
        exercise_words = ["出题", "练习", "quiz", "exercise", "question generation", "生成题"]
        mistake_words = ["错题", "错误", "mistake", "薄弱", "不会", "confusing", "confusion"]
        summary_words = ["总结", "profile", "学习画像", "复习计划", "study plan"]
        if any(w in q for w in exercise_words):
            return "exercise_agent"
        if any(w in q for w in mistake_words):
            return "mistake_agent"
        if any(w in q for w in summary_words):
            return "profile_agent"
        return "qa_agent"

    def _agent_instruction(self, mode: str, query: str) -> str:
        if mode == "exercise_agent":
            return f"""你是课程出题 Agent。请基于检索到的课程资料为学生生成高质量练习题。
学生需求：{query}
输出格式：
1. Knowledge Point
2. Difficulty
3. Problem
4. Hint
5. Reference Answer
6. Source Evidence
要求题目必须和资料相关，答案必须可验证。"""
        if mode == "mistake_agent":
            return f"""你是错题诊断 Agent。请分析学生问题中可能体现的概念混淆，并基于课程资料纠正。
学生问题：{query}
输出格式：
1. Possible Misconception
2. Correct Understanding
3. Step-by-step Fix
4. Similar Practice Question
5. Source Evidence"""
        if mode == "profile_agent":
            return f"""你是学习画像 Agent。请根据学生请求生成学习状态分析和复习建议。
学生请求：{query}
输出格式：
1. Recent Learning Focus
2. Weak Points
3. Recommended Review Plan
4. Next Questions to Practice"""
        return query

    def answer(self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = 5) -> AgentResult:
        mode = self.route(query)
        routed_query = self._agent_instruction(mode, query)
        result = self.rag.answer_question_with_sources(routed_query, chat_history=chat_history, top_k=top_k)
        return AgentResult(mode=mode, answer=result["answer"], sources=result["sources"])

    def answer_stream(self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = 5) -> Generator[str, None, None]:
        mode = self.route(query)
        yield f"[agent:{mode}]\n\n"
        routed_query = self._agent_instruction(mode, query)
        yield from self.rag.answer_question_stream(routed_query, chat_history=chat_history, top_k=top_k)
