import re
from typing import Dict, List
from tqdm import tqdm


class TextSplitter:
    """Semantic-aware text splitter.

    Compared with a naive fixed-length slicer, this splitter first cuts text into
    paragraph/sentence units, then merges neighbouring units under a target chunk
    size. It preserves sentence boundaries as much as possible and still keeps a
    configurable overlap window for retrieval continuity.
    """

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
        # keep paragraph boundaries, but remove excessive whitespace inside lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)

    def _split_long_unit(self, unit: str) -> List[str]:
        """Fallback split for a single sentence/paragraph longer than chunk_size."""
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
        return [p for p in pieces if p]

    def _sentence_units(self, text: str) -> List[str]:
        """Split text into semantic units: headings/paragraphs/sentences."""
        text = self._normalize_text(text)
        if not text:
            return []

        units: List[str] = []
        paragraphs = re.split(r"\n{2,}|(?<=--- 第 \d+ 页 ---)\n|(?<=--- 幻灯片 \d+ ---)\n", text)
        sentence_pattern = re.compile(r"[^。！？!?；;\.\n]+[。！？!?；;\.]?|[^\n]+")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Keep short headings or page/slide marks as independent units.
            if len(para) <= 80 and (para.startswith("---") or para.endswith("：") or para.endswith(":")):
                units.append(para)
                continue
            for match in sentence_pattern.finditer(para):
                sent = match.group(0).strip()
                if sent:
                    units.extend(self._split_long_unit(sent))
        return units

    def _tail_overlap(self, chunk: str) -> str:
        """Generate overlap text from the tail of the previous chunk."""
        if self.chunk_overlap <= 0 or len(chunk) <= self.chunk_overlap:
            return chunk if self.chunk_overlap > 0 else ""
        tail = chunk[-self.chunk_overlap:]
        # Prefer starting overlap at a sentence boundary when possible.
        boundary_positions = [tail.rfind(x) for x in ["。", "！", "？", ".", "!", "?", "\n"]]
        boundary = max(boundary_positions)
        if boundary >= 0 and boundary < len(tail) - 20:
            return tail[boundary + 1 :].strip()
        return tail.strip()

    def split_text(self, text: str) -> List[str]:
        """Split text with sentence-aware merging and overlap."""
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
                    # The overlap may make it too long; drop overlap before falling back.
                    chunks.extend(self._split_long_unit(unit))
                    current = ""
            else:
                chunks.extend(self._split_long_unit(unit))
                current = ""

        if current.strip():
            chunks.append(current.strip())

        # Remove exact duplicates caused by overlap edge cases while preserving order.
        deduped: List[str] = []
        seen = set()
        for chunk in chunks:
            key = re.sub(r"\s+", " ", chunk).strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(chunk)
        return deduped

    def split_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Split all supported documents.

        PDF/PPTX are still page/slide-aware, but each page/slide can now be split
        into semantic chunks when it is long. This keeps source attribution precise
        while avoiding overly large retrieval units.
        """
        chunks_with_metadata: List[Dict[str, str]] = []

        for doc in tqdm(documents, desc="处理文档", unit="文档"):
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
