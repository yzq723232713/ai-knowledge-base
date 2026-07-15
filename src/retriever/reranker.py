"""Reranker 精排模块"""
from typing import List
from sentence_transformers import CrossEncoder


class Reranker:
    """粗召回 → 精排，取 Top-K"""

    def __init__(self, model_name: str = None):
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        if model_name is None:
            # 优先从 ModelScope 缓存加载
            model_name = "C:/Users/qiang/.cache/modelscope/BAAI/bge-reranker-v2-m3"

        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """从 candidates 中精排取 Top-K → [(doc, score), ...]"""
        if not candidates:
            return []
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]