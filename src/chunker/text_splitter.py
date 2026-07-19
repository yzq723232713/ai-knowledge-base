"""文本切分器：递归降级切分"""
from typing import List
from src.models import Chunk


class TextSplitter:
    """递归降级切分：\\n\\n → \\n → 。→ 空格 → 硬切"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, source: str) -> List[Chunk]:
        pieces = self._recursive_split(text)
        chunks = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                chunks.append(Chunk(text=piece, index=len(chunks), source=source))
            else:
                for j in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                    chunks.append(Chunk(text=piece[j:j+self.chunk_size], index=len(chunks), source=source))
        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", "。", " "]
        return self._split_by_seps(text, separators)

    def _split_by_seps(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        sep = separators[0]
        if sep not in text:
            return self._split_by_seps(text, separators[1:])
        pieces = [p.strip() for p in text.split(sep) if p.strip()]
        if not pieces:
            return [text]
        if all(len(p) <= self.chunk_size for p in pieces):
            return pieces
        result = []
        for p in pieces:
            if len(p) <= self.chunk_size:
                result.append(p)
            else:
                result.extend(self._split_by_seps(p, separators[1:]))
        return result
    
    def stats(self, chunks: List[Chunk]) -> dict:
        """返回切分统计：总数、平均/最大/最小块大小"""
        sizes = [len(c.text) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": round(sum(sizes) / len(chunks), 1) if chunks else 0,
            "max_size": max(sizes) if chunks else 0,
            "min_size": min(sizes) if chunks else 0,
        }