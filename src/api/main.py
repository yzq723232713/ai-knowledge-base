import sys
sys.path.insert(0, "D:/ai-knowledge-base")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from pathlib import Path
import tempfile
from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter import TextSplitter
from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.generator.generator import Generator

app = FastAPI(title="企业知识库RAG问答系统", version="0.1.0")
loader = DocumentLoader()
splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
embedder = Embedder()
store = VectorStore(collection_name="api_kb")
generator = Generator()

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

class ChatResponse(BaseModel):
    question: str
    answer: str

class SearchRequest(BaseModel):
    query: str
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
        return {"status": "ok", "filename": file.filename, "chunks": len(chunks)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        
@app.post("/search")
async def search(req: SearchRequest):
    q_emb = embedder.embed_query(req.query).tolist()
    docs, metas, dists = store.search(q_emb, top_k=req.top_k)
    results = [{"text": d, "source": m.get("source", ""), "score": round(dist, 4)}
               for d, m, dist in zip(docs, metas, dists)]
    return {"query": req.query, "results": results}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    q_emb = embedder.embed_query(req.question).tolist()
    docs, metas, _ = store.search(q_emb, top_k=req.top_k)
    sources = [m.get("source", "") for m in metas]
    answer = generator.generate(req.question, docs, sources)
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
    q_emb = embedder.embed_query(req.question).tolist()
    docs, metas, _ = store.search(q_emb, top_k=req.top_k)
    sources = [m.get("source", "") for m in metas]
    async def event_stream():
        for chunk in generator.generate_stream(req.question, docs, sources):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
