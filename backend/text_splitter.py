import re
from typing import Dict, List

from tqdm import tqdm


class TextSplitter:
    """Semantic-aware text splitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _split_long_unit(self, unit: str) -> List[str]:
        if len(unit) <= self.chunk_size:
            return [unit]
        pieces: List[str] = []
        start = 0
        while start < len(unit):
            end = min(start + self.chunk_size, len(unit))
            pieces.append(unit[start:end].strip())
            if end >= len(unit):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return [piece for piece in pieces if piece]

    def _sentence_units(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        text = re.sub(r"(--- 第 \d+ 页 ---)", r"\1\n\n", text)
        text = re.sub(r"(--- 幻灯片 \d+ ---)", r"\1\n\n", text)
        paragraphs = re.split(r"\n{2,}", text)
        sentence_pattern = re.compile(r"[^。！？?.\n]+[。！？?.]?|[^\n]+")

        units: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= 80 and (para.startswith("---") or para.endswith("：") or para.endswith(":")):
                units.append(para)
                continue
            for match in sentence_pattern.finditer(para):
                sentence = match.group(0).strip()
                if sentence:
                    units.extend(self._split_long_unit(sentence))
        return units

    def _tail_overlap(self, chunk: str) -> str:
        if self.chunk_overlap <= 0 or len(chunk) <= self.chunk_overlap:
            return chunk if self.chunk_overlap > 0 else ""
        tail = chunk[-self.chunk_overlap:]
        boundary_positions = [tail.rfind(mark) for mark in ["。", "！", "？", ".", "!", "?", "\n"]]
        boundary = max(boundary_positions)
        if boundary >= 0 and boundary < len(tail) - 20:
            return tail[boundary + 1 :].strip()
        return tail.strip()

    def split_text(self, text: str) -> List[str]:
        units = self._sentence_units(text)
        if not units:
            return []

        chunks: List[str] = []
        current = ""

        for unit in units:
            candidate = (current + "\n" + unit).strip() if current else unit
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                overlap = self._tail_overlap(current)
                current = (overlap + "\n" + unit).strip() if overlap else unit
                if len(current) > self.chunk_size:
                    chunks.extend(self._split_long_unit(unit))
                    current = ""
            else:
                chunks.extend(self._split_long_unit(unit))
                current = ""

        if current.strip():
            chunks.append(current.strip())

        deduped: List[str] = []
        seen = set()
        for chunk in chunks:
            key = re.sub(r"\s+", " ", chunk).strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(chunk)
        return deduped

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        chunks_with_metadata: List[Dict[str, str]] = []

        for doc in tqdm(documents, desc="处理文档", unit="doc"):
            content = doc.get("content", "") or ""
            filetype = doc.get("filetype", "")
            chunks = self.split_text(content)
            if not chunks:
                continue

            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "content": chunk,
                    "filename": doc.get("filename", "unknown"),
                    "filepath": doc.get("filepath", ""),
                    "filetype": filetype,
                    "page_number": doc.get("page_number", 0),
                    "chunk_id": i,
                    "images": doc.get("images", []),
                }
                chunks_with_metadata.append(chunk_data)

        print(f"\n文档处理完成，共 {len(chunks_with_metadata)} 个语义块")
        return chunks_with_metadata
