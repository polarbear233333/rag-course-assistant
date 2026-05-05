from typing import Dict, Generator, List, Optional, Tuple

from openai import OpenAI

try:
    from .config import MAX_TOKENS, MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, TOP_K
    from .vector_store import VectorStore
except ImportError:
    from config import MAX_TOKENS, MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, TOP_K
    from vector_store import VectorStore


class RAGAgent:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        self.vector_store = VectorStore()
        self.system_prompt = """
你是《算法设计与分析》课程的智能助教。你必须优先基于检索到的课程资料回答问题，并严格区分“课程资料内容”和“课外扩展”。

核心规则：
1. 回答课程知识点时，必须引用资料出处，格式尽量使用“文件名 P页码 / chunk编号”。
2. 不允许伪造资料来源、页码、定理名称或课程结论。
3. 如果检索资料不足以直接回答，先说明“课程资料中未找到直接依据”，再给出课外扩展，并明确标注为课外扩展。
4. 回答应结构清晰：定义/直觉/算法步骤/复杂度/正确性要点/常见误区，按问题需要选择。
5. 如果学生要求“出题、生成习题、自动出题、作为练习题”，切换为出题模式，输出：题目、难度、参考资料、提示、标准参考答案。
6. 语气保持耐心、清晰、像真实助教，最后可以给一个引导性问题。
"""

    def _source_label(self, doc: Dict, idx: int) -> str:
        meta = doc.get("metadata", {}) or {}
        filename = meta.get("filename", "unknown")
        page_number = meta.get("page_number", "N/A")
        chunk_id = meta.get("chunk_id", "0")
        return f"[{idx}] {filename} P{page_number} / chunk {chunk_id}"

    def retrieve_context(self, query: str, top_k: int = TOP_K) -> Tuple[str, List[Dict]]:
        retrieved_docs = self.vector_store.search_hybrid(query, top_k=top_k)
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = self._source_label(doc, idx)
            rrf_score = doc.get("rrf_score")
            score_text = f"，RRF={rrf_score:.4f}" if isinstance(rrf_score, (int, float)) else ""
            context_parts.append(
                f"【资料片段 {idx}】\n来源：{source}{score_text}\n内容：\n{content}\n"
            )
        return "\n".join(context_parts), retrieved_docs

    def retrieve_sources(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Return retrieval results for frontend debug/source cards."""
        docs = self.vector_store.search_hybrid(query, top_k=top_k)
        sources = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {}) or {}
            content = doc.get("content", "") or ""
            debug = doc.get("debug", {}) or {}
            sources.append(
                {
                    "rank": idx,
                    "id": doc.get("id"),
                    "filename": meta.get("filename", "unknown"),
                    "page_number": meta.get("page_number", "N/A"),
                    "chunk_id": meta.get("chunk_id", "0"),
                    "filetype": meta.get("filetype", ""),
                    "rrf_score": doc.get("rrf_score"),
                    "dense_rank": debug.get("dense_rank"),
                    "bm25_rank": debug.get("bm25_rank"),
                    "preview": content[:360] + ("..." if len(content) > 360 else ""),
                }
            )
        return sources

    def _build_messages(self, query: str, context: str, chat_history: Optional[List[Dict]] = None) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            # Keep recent history only to avoid context explosion.
            messages.extend(chat_history[-8:])

        user_text = f"""请基于以下课程资料回答学生的问题。

【课程资料】
{context}

【学生问题】
{query}

回答要求：
- 优先使用课程资料中的内容；
- 引用时使用资料片段编号、文件名和页码；
- 如果资料不足，请明确说明并标注为课外扩展；
- 回答要结构化、适合学生理解。"""
        messages.append({"role": "user", "content": user_text})
        return messages

    def generate_response(self, query: str, context: str, chat_history: Optional[List[Dict]] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(query, context, chat_history),
                temperature=0.4,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def generate_response_stream(
        self, query: str, context: str, chat_history: Optional[List[Dict]] = None
    ) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(query, context, chat_history),
                temperature=0.4,
                max_tokens=MAX_TOKENS,
                stream=True,
            )
            yielded = False
            for chunk in stream:
                content_piece = None
                try:
                    choice = chunk.choices[0]
                    if getattr(choice, "delta", None) is not None:
                        content_piece = getattr(choice.delta, "content", None)
                    if not content_piece and getattr(choice, "message", None) is not None:
                        content_piece = getattr(choice.message, "content", None)
                except Exception:
                    content_piece = None
                if content_piece:
                    yielded = True
                    yield content_piece
            if not yielded:
                yield self.generate_response(query, context, chat_history)
        except Exception:
            yield self.generate_response(query, context, chat_history)

    def answer_question(self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K) -> str:
        context, _ = self.retrieve_context(query, top_k=top_k)
        if not context:
            context = "（未检索到特别相关的课程材料）"
        return self.generate_response(query, context, chat_history)

    def answer_question_with_sources(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> Dict:
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)
        if not context:
            context = "（未检索到特别相关的课程材料）"
        answer = self.generate_response(query, context, chat_history)
        return {"answer": answer, "sources": self.retrieve_sources(query, top_k=top_k), "raw_docs": retrieved_docs}

    def answer_question_stream(
        self, query: str, chat_history: Optional[List[Dict]] = None, top_k: int = TOP_K
    ) -> Generator[str, None, None]:
        context, _ = self.retrieve_context(query, top_k=top_k)
        if not context:
            context = "（未检索到特别相关的课程材料）"
        yield from self.generate_response_stream(query, context, chat_history)

    def chat(self) -> None:
        print("=" * 60)
        print("欢迎使用智能课程助教系统！")
        print("=" * 60)
        chat_history: List[Dict] = []
        while True:
            try:
                query = input("\n学生: ").strip()
                if not query:
                    continue
                answer = self.answer_question(query, chat_history=chat_history)
                print(f"\n助教: {answer}")
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")
