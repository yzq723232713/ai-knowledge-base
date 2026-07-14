"""Embedding 模块：文本 → 向量

提供两种向量化方案：
  Embedder    — 本地模型（bge-small，免费，离线，512 维）
  APIEmbedder — 通义千问 text-embedding-v3（需联网，1024 维）
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 方案一：本地 Embedding（bge-small-zh-v1.5，512 维）
# ============================================================

class Embedder:
    """本地向量化器。优先从缓存加载，缓存缺失时走镜像下载。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量向量化。返回 (n, dim) 的 numpy 数组。"""
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        """单个查询向量化。"""
        return self.model.encode(query, show_progress_bar=False)


# ============================================================
# 方案二：API Embedding（通义千问 text-embedding-v3，1024 维）
# ============================================================

class APIEmbedder:
    """API 向量化器。128 条一批，asyncio.Semaphore 控制并发，Key 自动从 .env 读取。"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        batch_size: int = 128,
        max_concurrency: int = 5,
    ):
        from openai import OpenAI
        import os

        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("DASHSCOPE_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.dim = 1024

    def embed(self, texts: List[str]) -> np.ndarray:
        return asyncio.run(self._embed_async(texts))

    async def async_embed(self, texts: List[str]) -> np.ndarray:
        return await self._embed_async(texts)

    def embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[query])
        return np.array(resp.data[0].embedding, dtype=np.float32)

    async def _embed_async(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []

        async def process_batch(batch: List[str]) -> List[List[float]]:
            async with self._semaphore:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                return [d.embedding for d in resp.data]

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        tasks = [process_batch(b) for b in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Embedding batch 失败: {r}")
                raise r
            all_embeddings.extend(r)

        return np.array(all_embeddings, dtype=np.float32)
