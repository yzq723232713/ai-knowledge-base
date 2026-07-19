"""向量存储模块：Chroma 封装，支持元数据过滤"""
from typing import List, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Chroma 向量库封装：入库、检索、元数据过滤"""

    def __init__(self, collection_name: str = "knowledge_base", persist_dir: str = None):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        # 每次重建（开发阶段方便调试，生产环境去掉 delete）
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(name=collection_name)
        self._next_id = 0

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict] = None,
        ids: List[str] = None,
    ):
        """批量入库。ids 不传则自动生成。"""
        if ids is None:
            ids = [f"chunk_{self._next_id + i}" for i in range(len(documents))]
            self._next_id += len(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> tuple:
        """
        向量检索。返回 (documents, metadatas, distances)。

        where 示例：{"source": "产品手册.pdf"}  或  {"page": {"$gte": 2}}
        """
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results["documents"][0], results["metadatas"][0], results["distances"][0]

    def count(self) -> int:
        return self.collection.count()