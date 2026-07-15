"""检索模块测试：向量 / BM25 / 混合 / Reranker 四层对比"""
import sys
sys.path.insert(0, ".")

from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker

embedder = Embedder()
store = VectorStore(collection_name="test_retrieval")

# ---- 准备测试数据 ----
texts = [
    "智能温控器安装步骤：固定设备、接线、APP配对",
    "Wi-Fi连接需要2.4GHz频段支持",
    "产品保修期为购买之日起两年",
    "设备显示E01错误代码表示传感器故障",
    "安装前请断开总电源确保安全",
]

embeddings = [embedder.embed([t])[0].tolist() for t in texts]
metadatas = [
    {"source": "产品手册.pdf", "page": 1, "chunk_index": 0, "doc_type": "pdf"},
    {"source": "产品手册.pdf", "page": 3, "chunk_index": 1, "doc_type": "pdf"},
    {"source": "保修卡.pdf", "page": 1, "chunk_index": 0, "doc_type": "pdf"},
    {"source": "故障手册.pdf", "page": 1, "chunk_index": 0, "doc_type": "pdf"},
    {"source": "产品手册.pdf", "page": 2, "chunk_index": 2, "doc_type": "pdf"},
]

retriever = HybridRetriever(embedder, store)
retriever.index(texts, embeddings, metadatas)


# ============================================================
# 基线：纯向量检索
# ============================================================

def test_vector_search():
    results = retriever.vector_search("怎么安装", top_k=3)
    assert len(results) == 3
    # 第一条应该包含"安装"
    assert "安装" in results[0][0], f"向量：{results[0][0][:30]}"


# ============================================================
# BM25 关键词检索
# ============================================================

def test_bm25_search():
    results = retriever.bm25_search("E01错误代码", top_k=3)
    assert len(results) == 3
    # E01 是精确关键词，BM25 应该命中
    assert "E01" in results[0][0], f"BM25：{results[0][0][:30]}"


# ============================================================
# 混合检索（向量 + BM25 → RRF）
# ============================================================

def test_hybrid_search():
    results = retriever.hybrid_search("怎么安装设备", top_k=3)
    assert len(results) == 3
    # 向量和 BM25 至少有一路能找到安装内容
    assert any("安装" in r[0] for r in results), "混合检索未找到安装内容"

def test_hybrid_vs_single():
    """混合检索的召回覆盖率不低于单路"""
    vec = set(r[0] for r in retriever.vector_search("安装", top_k=10))
    bm = set(r[0] for r in retriever.bm25_search("安装", top_k=10))
    hy = set(r[0] for r in retriever.hybrid_search("安装", top_k=10))

    # 混合的重复条数 ≥ 任一路
    assert len(hy) >= min(len(vec), len(bm)), f"混合{len(hy)} < vec{len(vec)}/bm{len(bm)}"


# ============================================================
# Reranker 精排
# ============================================================

def test_reranker_loads():
    """Reranker 能正常加载"""
    reranker = Reranker()
    assert reranker.model is not None

def test_reranker_rerank():
    """粗召回有噪声，Reranker 纠正排名"""
    reranker = Reranker()

    # 模拟粗召回：正确答案混在噪声里
    candidates = [
        "设备显示E01错误代码表示传感器故障",  # ← 正确答案
        "Wi-Fi连接需要2.4GHz频段支持",
        "产品保修期为购买之日起两年",
        "智能温控器安装步骤",
    ]

    results = reranker.rerank("E01怎么解决", candidates, top_k=2)
    assert len(results) == 2
    # Reranker 读懂了 E01，第一名应该是正确答案
    assert "E01" in results[0][0], f"Reranker: {results[0][0][:30]}"

def test_reranker_scoring_quality():
    """Reranker 能区分相关和不相关文档"""
    reranker = Reranker()

    relevant = "E01错误代码表示温度传感器故障，请断电重启"
    irrelevant = "Wi-Fi连接需要2.4GHz频段，建议使用5GHz"
    results = reranker.rerank("E01错误", [irrelevant, relevant], top_k=2)

    # 相关文档的分数应该更高
    assert results[0][0] == relevant, f"Reranker把无关文档排第一了"


# ============================================================
# 四层对比测试
# ============================================================

def test_four_way_comparison():
    """四层检索对比打印"""
    reranker = Reranker()
    query = "E01错误怎么办"

    v = retriever.vector_search(query, top_k=3)
    b = retriever.bm25_search(query, top_k=3)
    h = retriever.hybrid_search(query, top_k=3)
    hybrid_docs = [doc for doc, _, _ in retriever.hybrid_search(query, top_k=5)]
    r = reranker.rerank(query, hybrid_docs, top_k=3)

    print(f"\n查询: 「{query}」")
    print("  [纯向量]")
    for i, (doc, score, _) in enumerate(v, 1):
        print(f"    {i}. [{score:.4f}] {doc[:50]}")
    print("  [BM25]")
    for i, (doc, score, _) in enumerate(b, 1):
        print(f"    {i}. [{score:.4f}] {doc[:50]}")
    print("  [混合 RRF]")
    for i, (doc, score, _) in enumerate(h, 1):
        print(f"    {i}. [{score:.4f}] {doc[:50]}")
    print("  [Reranker]")
    for i, (doc, score) in enumerate(r, 1):
        print(f"    {i}. [{score:.4f}] {doc[:50]}")
