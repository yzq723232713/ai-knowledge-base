import sys
sys.path.insert(0, "D:/ai-knowledge-base")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any
from pathlib import Path
import tempfile
from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter import TextSplitter
from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker
from src.generator.generator import Generator

app = FastAPI(title="企业知识库RAG问答系统", version="0.1.0")
loader = DocumentLoader()
splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
embedder = Embedder()
store = VectorStore(collection_name="api_kb")
hybrid = HybridRetriever(embedder, store)
reranker = Reranker()
generator = Generator()

# 维护已上传文档数据，用于重建 Hybrid 索引
_doc_texts: list[str] = []
_doc_embeddings: list[list[float]] = []
_doc_metas: list[dict] = []

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

class ChatResponse(BaseModel):
    question: str
    answer: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class MultiChatRequest(BaseModel):
    question: str
    messages: list[dict] = []  # [{"role":"user/assistant","content":"..."}]
    top_k: int = 5

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        doc = loader.load(tmp_path)
        if not doc.content:
            raise HTTPException(400, f"无法解析文件: {file.filename}")
        chunks = splitter.split(doc.content, source=file.filename)
        texts = [c.text for c in chunks]
        vecs = embedder.embed(texts).tolist()
        metas = [{"source": c.source, "chunk_index": c.index} for c in chunks]
        store.add(documents=texts, embeddings=vecs, metadatas=metas)
        _doc_texts.extend(texts)
        _doc_embeddings.extend(vecs)
        _doc_metas.extend(metas)
        hybrid.index(_doc_texts, _doc_embeddings, _doc_metas)
        return {"status": "ok", "filename": file.filename, "chunks": len(chunks)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        
def _retrieve(query: str, top_k: int) -> tuple[list[str], list[str]]:
    """统一检索入口：HybridRetriever 粗召 → Reranker 精排"""
    if not _doc_texts:
        return [], []
    coarse_k = min(top_k * 3, len(_doc_texts))
    results = hybrid.hybrid_search(query, top_k=coarse_k)
    if not results:
        return [], []
    cands = [r[0] for r in results]
    ranked = reranker.rerank(query, cands, top_k=top_k)
    docs = [r[0] for r in ranked]
    # 反查来源
    sources = []
    for d in docs:
        try:
            idx = _doc_texts.index(d)
            sources.append(_doc_metas[idx].get("source", ""))
        except ValueError:
            sources.append("未知")
    return docs, sources

@app.post("/search")
async def search(req: SearchRequest):
    docs, sources = _retrieve(req.query, req.top_k)
    results = [{"text": d, "source": s, "score": 0.0}
               for d, s in zip(docs, sources)]
    return {"query": req.query, "results": results}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    docs, sources = _retrieve(req.question, req.top_k)
    answer = generator.generate(req.question, docs, sources)
    return ChatResponse(question=req.question, answer=answer)

@app.post("/chat/multi", response_model=ChatResponse)
async def chat_multi(req: MultiChatRequest):
    docs, sources = _retrieve(req.question, req.top_k)
    answer = generator.generate_with_history(
        req.question, docs, sources, history=req.messages or None,
    )
    return ChatResponse(question=req.question, answer=answer)

@app.get("/documents")
async def list_documents():
    n = store.count()
    if n == 0:
        return {"documents": []}
    docs, metas, _ = store.search(embedder.embed_query("汇总").tolist(), top_k=n)
    seen = {}
    for m in metas:
        src = m.get("source", "未知")
        seen[src] = seen.get(src, 0) + 1
    return {"documents": [{"source": k, "chunks": v} for k, v in seen.items()]}

@app.get("/health")
async def health():
    return {"status": "ok"}
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    docs, sources = _retrieve(req.question, req.top_k)
    async def event_stream():
        for chunk in generator.generate_stream(req.question, docs, sources):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/chat/multi/stream")
async def chat_multi_stream(req: MultiChatRequest):
    docs, sources = _retrieve(req.question, req.top_k)
    async def event_stream():
        for chunk in generator.generate_stream_with_history(
            req.question, docs, sources, history=req.messages or None,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
