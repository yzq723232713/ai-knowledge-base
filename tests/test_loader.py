"""文档加载器测试"""
import sys
sys.path.insert(0, ".")

from src.loader.document_loader import DocumentLoader

def test_load_txt():
    doc = DocumentLoader.load("data/test_docs/产品说明.txt")
    assert len(doc.content) > 0
    assert doc.source == "产品说明.txt"
    assert doc.file_type == "txt"

def test_load_pdf():
    doc = DocumentLoader.load("data/test_docs/会议纪要.pdf")
    assert len(doc.content) > 0
    assert doc.file_type == "pdf"

def test_load_docx():
    doc = DocumentLoader.load("data/test_docs/需求文档.docx")
    assert len(doc.content) > 0
    assert doc.file_type == "docx"

def test_load_csv():
    doc = DocumentLoader.load("data/test_docs/员工信息.csv")
    assert len(doc.content) > 0
    assert doc.file_type == "csv"

def test_damaged_file_does_not_crash():
    """损坏文件不崩溃，返回空 content"""
    doc = DocumentLoader.load("data/test_docs/损坏文件.pdf")
    assert doc.content == ""
    assert doc.source == "损坏文件.pdf"

def test_unsupported_format():
    doc = DocumentLoader.load("data/test_docs/readme.md")
    assert doc.file_type == "unknown"

def test_batch_load():
    """批量加载 4 种格式，统一返回 Document"""
    files = [
        "data/test_docs/产品说明.txt",
        "data/test_docs/会议纪要.pdf",
        "data/test_docs/员工信息.csv",
        "data/test_docs/需求文档.docx",
    ]
    docs = [DocumentLoader.load(f) for f in files]
    assert all(d.file_type in {"txt", "pdf", "csv", "docx"} for d in docs)