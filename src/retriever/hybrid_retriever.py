"""混合检索器：向量检索 + BM25 关键词检索 → RRF 融合"""
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore


class HybridRetriever:
    """向量 + BM25 多路召回，RRF 融合"""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self.documents: List[str] = []          # BM25 索引用
        self.doc_ids: List[str] = []            # Chroma ID 映射
        self.metadatas: List[dict] = []         # 元数据
        self.bm25: BM25Okapi = None

    def index(self, documents: List[str], embeddings: List[List[float]],
              metadatas: List[dict] = None, ids: List[str] = None):
        """建索引：向量入库 + BM25 分词索引"""
        self.store.add(documents=documents, embeddings=embeddings,
                       metadatas=metadatas, ids=ids)

        # BM25 索引
        self.documents = documents
        self.metadatas = metadatas or [{}] * len(documents)
        self.doc_ids = ids or [f"chunk_{i}" for i in range(len(documents))]
        tokenized = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, top_k: int = 20) -> List[tuple]:
        """纯向量检索 → [(doc_text, score, index), ...]"""
        q_emb = self.embedder.embed_query(query).tolist()
        docs, metas, dists = self.store.search(q_emb, top_k=top_k)
        results = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            idx = self.documents.index(doc) if doc in self.documents else -1
            results.append((doc, -dist, idx))
        return results

    def bm25_search(self, query: str, top_k: int = 20) -> List[tuple]:
        """纯 BM25 关键词检索"""
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], float(scores[i]), i) for i in top_indices]

    def hybrid_search(self, query: str, top_k: int = 10, k: int = 60) -> List[tuple]:
        """混合检索：向量 + BM25 → RRF 融合"""
        vec_results = self.vector_search(query, top_k=20)
        bm_results = self.bm25_search(query, top_k=20)

        rrf_scores = {}
        for rank, (_, _, idx) in enumerate(vec_results, 1):
            if idx >= 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        for rank, (_, _, idx) in enumerate(bm_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)

        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [(self.documents[i], rrf_scores[i], i) for i in sorted_indices]