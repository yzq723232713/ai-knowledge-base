"""Day 37：端到端集成测试 — 全流程验证"""
import sys
sys.path.insert(0, "E:/ai-knowledge-base")

from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter import TextSplitter
from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.generator.generator import Generator
from pathlib import Path

loader = DocumentLoader()
splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
embedder = Embedder()

# ============================================================
# 测试 1：完整流水线（加载->切块->向量化->入库->检索->生成）
# ============================================================

def test_full_pipeline():
    """端到端：PDF上传 -> 一切 -> 问答"""
    store = VectorStore(collection_name="test_e2e")
    generator = Generator()

    # Step 1-2: 加载 + 切块
    test_files = list(Path("E:/learn/day15/test_docs").iterdir())
    all_chunks = []
    for f in test_files:
        if f.suffix.lower() not in {".pdf", ".docx", ".txt", ".csv"}:
            continue
        doc = loader.load(str(f))
        assert doc.content, f"{f.name} 加载失败"
        chunks = splitter.split(doc.content, source=doc.source)
        assert len(chunks) > 0, f"{f.name} 切分为空"
        all_chunks.extend(chunks)

    assert len(all_chunks) >= 4, f"至少 4 个块，实际 {len(all_chunks)}"

    # Step 3-4: 向量化 + 入库
    texts = [c.text for c in all_chunks]
    vecs = embedder.embed(texts).tolist()
    metas = [{"source": c.source, "chunk_index": c.index} for c in all_chunks]
    store.add(documents=texts, embeddings=vecs, metadatas=metas)
    assert store.count() == len(all_chunks)

    # Step 5-6: 检索 + 生成
    queries = [
        "会议的主题是什么",
        "怎么安装设备",
        "E01错误代码怎么解决",
        "张三在哪个部门",
        "产品有哪些功能",
    ]

    for q in queries:
        q_emb = embedder.embed_query(q).tolist()
        docs, metas, _ = store.search(q_emb, top_k=3)
        sources = [m.get("source", "") for m in metas]
        answer = generator.generate(q, docs, sources)

        assert len(answer) > 0, f"{q} 生成失败"
        print(f"\n  问题: {q}")
        for i, (d, s) in enumerate(zip(docs, sources)):
            print(f"    检索{i}: [{s}] {d[:60]}...")
        print(f"    回答: {answer[:100]}...")

    print(f"\n  全部 {len(queries)} 个问题端到端通过")
