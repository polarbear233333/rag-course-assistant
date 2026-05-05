from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROFILE_PATH = Path("learning_profile.json")

KEYWORDS = {
    "dynamic_programming": ["dp", "dynamic programming", "动态规划", "状态转移", "最优子结构"],
    "graph_algorithms": ["dijkstra", "bellman", "mst", "最短路", "最小生成树", "图"],
    "greedy": ["greedy", "贪心", "exchange argument", "交换论证"],
    "divide_and_conquer": ["divide", "master theorem", "分治", "主定理"],
    "complexity": ["complexity", "big-o", "复杂度", "渐进"],
    "proof": ["证明", "correctness", "invariant", "归纳", "不变式"],
}


def load_profile() -> Dict:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"total_questions": 0, "topics": {}, "recent_questions": [], "updated_at": None}


def _detect_topics(text: str) -> List[str]:
    low = text.lower()
    topics = []
    for topic, kws in KEYWORDS.items():
        if any(k.lower() in low for k in kws):
            topics.append(topic)
    return topics or ["general_algorithm"]


def update_profile(question: str, answer: str = "", mode: str = "qa_agent") -> Dict:
    profile = load_profile()
    topics = _detect_topics(question + " " + answer[:300])
    profile["total_questions"] = int(profile.get("total_questions", 0)) + 1
    topic_counts = Counter(profile.get("topics", {}))
    for t in topics:
        topic_counts[t] += 1
    profile["topics"] = dict(topic_counts.most_common())
    item = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "question": question,
        "topics": topics,
    }
    recent = profile.get("recent_questions", []) + [item]
    profile["recent_questions"] = recent[-50:]
    profile["updated_at"] = item["time"]
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def profile_summary() -> Dict:
    profile = load_profile()
    topics = profile.get("topics", {})
    total = max(int(profile.get("total_questions", 0)), 1)
    distribution = [{"topic": k, "count": v, "ratio": round(v / total, 3)} for k, v in topics.items()]
    weak = [x["topic"] for x in distribution[-3:]] if distribution else []
    strong = [x["topic"] for x in distribution[:3]] if distribution else []
    return {**profile, "topic_distribution": distribution, "strong_topics": strong, "review_candidates": weak}
