"""向量存储模块测试"""
import sys
sys.path.insert(0, ".")

from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
import numpy as np

embedder = Embedder()

def test_add_and_count():
    store = VectorStore(collection_name="test_add")
    texts = ["文档A", "文档B", "文档C"]
    vecs = embedder.embed(texts).tolist()
    store.add(documents=texts, embeddings=vecs)
    assert store.count() == 3

def test_search_returns_correct_count():
    store = VectorStore(collection_name="test_search")
    texts = ["智能温控器安装步骤", "Wi-Fi连接问题", "保修条款说明"]
    vecs = embedder.embed(texts).tolist()
    metas = [{"source": "产品手册.pdf"}, {"source": "产品手册.pdf"}, {"source": "保修卡.pdf"}]

    store.add(documents=texts, embeddings=vecs, metadatas=metas)

    q_emb = embedder.embed_query("设备怎么装").tolist()
    docs, metas_out, dists = store.search(q_emb, top_k=3)

    assert len(docs) == 3
    assert len(dists) == 3

def test_search_relevance():
    """安装类查询，安装步骤应排第一"""
    store = VectorStore(collection_name="test_relevance")
    texts = ["安装步骤：固定、接线、配对", "保修期两年", "Wi-Fi需要2.4GHz"]
    vecs = embedder.embed(texts).tolist()
    store.add(documents=texts, embeddings=vecs)

    q_emb = embedder.embed_query("怎么安装").tolist()
    docs, _, _ = store.search(q_emb, top_k=3)

    assert "安装" in docs[0], f"第一个应该是安装内容，实际是: {docs[0][:30]}"

def test_metadata_filter():
    """元数据过滤：只看保修卡.pdf"""
    store = VectorStore(collection_name="test_filter")
    texts = ["安装步骤", "Wi-Fi设置", "保修两年"]
    vecs = embedder.embed(texts).tolist()
    metas = [
        {"source": "产品手册.pdf", "page": 1},
        {"source": "产品手册.pdf", "page": 3},
        {"source": "保修卡.pdf", "page": 1},
    ]
    store.add(documents=texts, embeddings=vecs, metadatas=metas)

    q_emb = embedder.embed_query("保修").tolist()

    # 不过滤 → 3 条
    docs_all, _, _ = store.search(q_emb, top_k=5)
    assert len(docs_all) == 3

    # 过滤 source → 只 1 条
    docs_filtered, _, _ = store.search(q_emb, top_k=5, where={"source": "保修卡.pdf"})
    assert len(docs_filtered) == 1
    assert "保修" in docs_filtered[0]

def test_metadata_schema():
    """元数据 Schema 验证：每个块四种标签"""
    store = VectorStore(collection_name="test_schema")
    texts = ["文档内容A", "文档内容B"]
    vecs = embedder.embed(texts).tolist()
    metas = [
        {"source": "产品手册.pdf", "page": 1, "chunk_index": 0, "doc_type": "pdf"},
        {"source": "产品手册.pdf", "page": 2, "chunk_index": 1, "doc_type": "pdf"},
    ]
    store.add(documents=texts, embeddings=vecs, metadatas=metas)

    q_emb = embedder.embed_query("文档").tolist()
    _, metas_out, _ = store.search(q_emb, top_k=2)

    # 每条元数据必须包含四种标签
    for m in metas_out:
        assert "source" in m
        assert "page" in m
        assert "chunk_index" in m
        assert "doc_type" in m
    print(f"  元数据 Schema 验证通过: {metas_out}")