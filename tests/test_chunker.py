"""文本切分器测试"""
import sys
sys.path.insert(0, ".")

from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter import TextSplitter
from pathlib import Path

def test_split_single_document():
    """单文档切分"""
    doc = DocumentLoader.load("data/test_docs/产品说明.txt")
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split(doc.content, source=doc.source)

    assert len(chunks) > 0
    assert all(c.source == "产品说明.txt" for c in chunks)
    assert all(len(c.text) <= 100 for c in chunks)

    s = splitter.stats(chunks)
    print(f"  产品说明.txt → {s['total_chunks']}块, 平均{s['avg_size']}字")

def test_batch_split_all_docs():
    """批量切分所有测试文档，统计总块数"""
    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []
    doc_stats = []

    for f in Path("data/test_docs").iterdir():
        if f.suffix.lower() not in {".pdf", ".docx", ".txt", ".csv"}:
            continue
        doc = DocumentLoader.load(str(f))
        if not doc.content:
            continue
        chunks = splitter.split(doc.content, source=doc.source)
        all_chunks.extend(chunks)
        s = splitter.stats(chunks)
        doc_stats.append({"source": doc.source, "chunks": len(chunks), "avg_size": s["avg_size"]})

    # 整体统计
    sizes = [len(c.text) for c in all_chunks]
    total = len(all_chunks)

    print(f"\n  总计: {total} 块")
    print(f"  平均块大小: {sum(sizes)/total:.1f} 字")
    print(f"  最大块: {max(sizes)} 字, 最小块: {min(sizes)} 字")

    for ds in doc_stats:
        print(f"    {ds['source']}: {ds['chunks']}块, 平均{ds['avg_size']}字")

    assert total > 0
    assert max(sizes) <= 300

def test_overlap():
    """验证块之间有重叠"""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=30)
    text = "A" * 250  # 250 个 A，会被切成多块
    chunks = splitter.split(text, source="test.txt")

    # 块1的结尾应该跟块2的开头有重叠
    assert len(chunks) >= 2
    end_of_first = chunks[0].text[-20:]
    start_of_second = chunks[1].text[:20]
    assert end_of_first in chunks[1].text  # 结尾内容出现在下一块中

def test_chunk_metadata_inheritance():
    """每个块的 source 继承自原文档"""
    doc = DocumentLoader.load("data/test_docs/产品说明.txt")
    splitter = TextSplitter(chunk_size=100)
    chunks = splitter.split(doc.content, source=doc.source)

    for i, c in enumerate(chunks):
        assert c.source == doc.source
        assert c.index == i  # 索引连续