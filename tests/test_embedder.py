"""Embedding 模块测试 — 本地 Embedder + APIEmbedder"""
import sys
sys.path.insert(0, ".")

from src.embedder.embedder import Embedder, APIEmbedder
import numpy as np

# ---- 模块级实例 ----
embedder = Embedder()
api_embedder = APIEmbedder()  # Key 自动从 .env 读取


# ============================================================
# 本地 Embedder 测试
# ============================================================

def test_local_dimension():
    """bge-small 输出 512 维"""
    assert embedder.dim == 512


def test_local_embed_query_shape():
    """单条查询 → (512,)"""
    vec = embedder.embed_query("测试")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_local_embed_batch_shape():
    """批量 3 条 → (3, 512)"""
    vecs = embedder.embed(["a", "b", "c"])
    assert vecs.shape == (3, 512)


def test_local_semantic_similarity():
    """语义相近 → 余弦相似度高；语义无关 → 低"""
    def cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    v1 = embedder.embed_query("智能温控器怎么安装")
    v2 = embedder.embed_query("设备安装步骤")
    v3 = embedder.embed_query("今天天气很好")

    sim = cos(v1, v2)
    diff = cos(v1, v3)
    assert sim > diff, f"相似({sim:.4f})应大于无关({diff:.4f})"


# ============================================================
# APIEmbedder 测试
# ============================================================

def test_api_dimension():
    """DeepSeek Embedding 输出 1024 维"""
    assert api_embedder.dim == 1024


def test_api_embed_query_shape():
    vec = api_embedder.embed_query("测试")
    assert vec.shape == (1024,)


def test_api_embed_batch():
    """3 条 → (3, 1024)"""
    vecs = api_embedder.embed(["a", "b", "c"])
    assert vecs.shape == (3, 1024)


def test_api_batch_size_split():
    """传 200 条 → 分成 128 + 72 两批"""
    texts = [f"文档{i}" for i in range(200)]
    vecs = api_embedder.embed(texts)
    assert vecs.shape == (200, 1024)


def test_api_semantic_similarity():
    """API 版也得能区分语义"""
    def cos(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    v1 = api_embedder.embed_query("智能温控器怎么安装")
    v2 = api_embedder.embed_query("设备安装步骤")
    v3 = api_embedder.embed_query("今天天气很好")

    sim = cos(v1, v2)
    diff = cos(v1, v3)
    assert sim > diff, f"相似({sim:.4f})应大于无关({diff:.4f})"
