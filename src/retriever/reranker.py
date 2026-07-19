"""Reranker 精排模块"""
from typing import List
from sentence_transformers import CrossEncoder


class Reranker:
    """粗召回 → 精排，取 Top-K"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            # 优先从 ModelScope 缓存加载；没有则自动下载
            model_name = "BAAI/bge-reranker-v2-m3"

        # 尝试本地缓存
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            # 缓存缺失 → 通过 ModelScope 下载到 ~/.cache/modelscope/
            from modelscope import snapshot_download
            model_dir = snapshot_download("BAAI/bge-reranker-v2-m3")
            self.model = CrossEncoder(model_dir)

    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """从 candidates 中精排取 Top-K → [(doc, score), ...]"""
        if not candidates:
            return []
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
