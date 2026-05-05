import hashlib
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import chromadb
import jieba
from chromadb.config import Settings
from nltk.tokenize import word_tokenize
from openai import OpenAI
from tqdm import tqdm

try:
    from .config import (
        BM25_B, BM25_K1, COLLECTION_NAME, OPENAI_API_BASE, OPENAI_API_KEY,
        OPENAI_EMBEDDING_MODEL, RRF_K, TOP_K, VECTOR_DB_PATH,
    )
except ImportError:
    from config import (
        BM25_B, BM25_K1, COLLECTION_NAME, OPENAI_API_BASE, OPENAI_API_KEY,
        OPENAI_EMBEDDING_MODEL, RRF_K, TOP_K, VECTOR_DB_PATH,
    )


class VectorStore:
    """Chroma vector store + in-memory BM25 + RRF hybrid retrieval."""

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        api_key: str = OPENAI_API_KEY,
        api_base: str = OPENAI_API_BASE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"description": "课程材料向量数据库"}
        )

        self._bm25_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._bm25_doc_len: Dict[str, int] = {}
        self._bm25_avgdl = 0.0
        self._bm25_built = False
        self._id_to_doc: Dict[str, Dict] = {}

    def _reset_bm25_cache(self) -> None:
        self._bm25_index = defaultdict(dict)
        self._bm25_doc_len = {}
        self._bm25_avgdl = 0.0
        self._bm25_built = False
        self._id_to_doc = {}

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取 embedding 失败: {e}")
            return []

    def _make_doc_id(self, chunk: Dict, idx: int) -> str:
        raw = "|".join(
            [
                str(chunk.get("filename", "doc")),
                str(chunk.get("page_number", 0)),
                str(chunk.get("chunk_id", idx)),
                chunk.get("content", "")[:80],
            ]
        )
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
        safe_name = re.sub(r"[^0-9A-Za-z_\-\.]+", "_", str(chunk.get("filename", "doc")))
        return f"{safe_name}_{chunk.get('page_number', 0)}_{chunk.get('chunk_id', idx)}_{digest}"

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        ids, documents, metadatas, embeddings = [], [], [], []

        for idx, chunk in enumerate(tqdm(chunks, desc="生成 embedding 并添加到向量库", unit="块")):
            content = (chunk.get("content", "") or "").strip()
            if not content:
                continue
            embedding = self.get_embedding(content)
            if not embedding:
                continue

            metadata = {
                "filename": chunk.get("filename", "unknown"),
                "filepath": chunk.get("filepath", ""),
                "filetype": chunk.get("filetype", ""),
                "page_number": str(chunk.get("page_number", 0)),
                "chunk_id": str(chunk.get("chunk_id", 0)),
            }
            doc_id = self._make_doc_id(chunk, idx)
            ids.append(doc_id)
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)

        if ids:
            # upsert makes repeated processing safer than add.
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            self._reset_bm25_cache()
            print(f"\n已将 {len(ids)} 个文档块写入向量数据库")

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        tokens: List[str] = []

        for part in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend([w.strip() for w in jieba.lcut(part) if w.strip()])

        english_parts = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
        if english_parts:
            joined = " ".join(english_parts)
            try:
                eng_tokens = word_tokenize(joined)
            except LookupError:
                eng_tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", joined)
            tokens.extend([w.strip() for w in eng_tokens if w.strip()])

        if not tokens:
            tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
        return [t.lower() for t in tokens]

    def build_bm25_index(self) -> None:
        if self._bm25_built:
            return
        self._reset_bm25_cache()

        try:
            data = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print("BM25 索引构建失败:", repr(e))
            self._bm25_built = True
            return

        ids = data.get("ids", []) if data else []
        docs = data.get("documents", []) if data else []
        metas = data.get("metadatas", []) if data else []
        if not ids:
            print("BM25 索引构建：向量库为空")
            self._bm25_built = True
            return

        total_len = 0
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            self._id_to_doc[doc_id] = {"id": doc_id, "content": content, "metadata": meta}
            tokens = self._tokenize(content)
            tf = Counter(tokens)
            doc_len = max(sum(tf.values()), 1)
            self._bm25_doc_len[doc_id] = doc_len
            total_len += doc_len
            for term, freq in tf.items():
                self._bm25_index[term][doc_id] = freq

        self._bm25_avgdl = total_len / max(len(ids), 1)
        self._bm25_built = True
        print(f"BM25 索引已构建：{len(ids)} 个文档块，平均长度 {self._bm25_avgdl:.1f}")

    def search_bm25(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        self.build_bm25_index()
        tokens = self._tokenize(query)
        if not tokens or not self._id_to_doc:
            return []

        scores = defaultdict(float)
        N = len(self._bm25_doc_len)
        for term in tokens:
            postings = self._bm25_index.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in postings.items():
                dl = self._bm25_doc_len.get(doc_id, 1)
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / (self._bm25_avgdl or 1.0))
                scores[doc_id] += idf * (tf * (BM25_K1 + 1)) / (denom or 1.0)

        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
        results = []
        for rank, doc_id in enumerate(ranked_ids, 1):
            item = dict(self._id_to_doc.get(doc_id, {}))
            item["score"] = float(scores[doc_id])
            item["rank"] = rank
            item["retrieval"] = "bm25"
            results.append(item)
        return results

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"向量检索失败: {e}")
            return []

        formatted_results: List[Dict] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for i, doc in enumerate(docs):
            distance: Optional[float] = distances[i] if i < len(distances) else None
            formatted_results.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distance,
                    "score": None if distance is None else 1.0 / (1.0 + float(distance)),
                    "rank": i + 1,
                    "retrieval": "dense",
                }
            )
        return formatted_results

    def search_hybrid(self, query: str, top_k: int = TOP_K, candidate_k: Optional[int] = None) -> List[Dict]:
        """Hybrid retrieval: dense vector + BM25, fused by Reciprocal Rank Fusion."""
        candidate_k = candidate_k or max(top_k * 3, 10)
        dense = self.search(query, candidate_k)
        sparse = self.search_bm25(query, candidate_k)

        scores = defaultdict(float)
        by_id: Dict[str, Dict] = {}
        debug = defaultdict(lambda: {"dense_rank": None, "bm25_rank": None, "dense_score": None, "bm25_score": None})

        for rank, item in enumerate(dense, 1):
            doc_id = item.get("id") or f"dense_{rank}"
            scores[doc_id] += 1.0 / (RRF_K + rank)
            by_id[doc_id] = item
            debug[doc_id]["dense_rank"] = rank
            debug[doc_id]["dense_score"] = item.get("score")

        for rank, item in enumerate(sparse, 1):
            doc_id = item.get("id") or f"bm25_{rank}"
            scores[doc_id] += 1.0 / (RRF_K + rank)
            by_id.setdefault(doc_id, item)
            debug[doc_id]["bm25_rank"] = rank
            debug[doc_id]["bm25_score"] = item.get("score")

        fused_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
        results: List[Dict] = []
        for rank, doc_id in enumerate(fused_ids, 1):
            item = dict(by_id.get(doc_id, {}))
            item["id"] = doc_id
            item["rank"] = rank
            item["retrieval"] = "hybrid_rrf"
            item["rrf_score"] = float(scores[doc_id])
            item["debug"] = debug[doc_id]
            results.append(item)
        return results

    def clear_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name, metadata={"description": "课程向量数据库"}
        )
        self._reset_bm25_cache()
        print("向量数据库已清空")

    def get_collection_count(self) -> int:
        return self.collection.count()
