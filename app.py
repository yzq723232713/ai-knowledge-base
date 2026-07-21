"""Day 38：Streamlit 极简 UI — 文件上传 + RAG 问答 + 引用来源"""
import sys, os
os.environ["HF_HUB_DISABLE_IMPORT_ERROR"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "fatal"
sys.path.insert(0, "D:/ai-knowledge-base")

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile
from pathlib import Path
from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter import TextSplitter
from src.embedder.embedder import Embedder
from src.retriever.vector_store import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker
from src.generator.generator import Generator

# ===== 初始化 =====
@st.cache_resource
def init_components():
    return {
        "loader": DocumentLoader(),
        "splitter": TextSplitter(chunk_size=300, chunk_overlap=50),
        "embedder": Embedder(),
        "store": VectorStore(collection_name="streamlit_kb"),
        "hybrid": HybridRetriever(
            Embedder(), VectorStore(collection_name="streamlit_kb")
        ),
        "reranker": Reranker(),
        "generator": Generator(),
    }

comp = init_components()
loader = comp["loader"]
splitter = comp["splitter"]
embedder = comp["embedder"]
store = comp["store"]
hybrid = comp["hybrid"]
hybrid.store = store  # 确保 hybrid 和 store 用同一个 collection
reranker = comp["reranker"]
generator = comp["generator"]

# 文档数据缓存（用于重建 Hybrid 索引）
if "_doc_texts" not in st.session_state:
    st.session_state._doc_texts = []
    st.session_state._doc_embeddings = []
    st.session_state._doc_metas = []

# ===== UI =====
st.set_page_config(page_title="知识库问答", layout="wide")
st.title("📚 企业知识库 RAG 问答")

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("📤 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / DOCX / TXT / CSV",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "csv"],
    )
    if uploaded_files and st.button("开始入库"):
        for f in uploaded_files:
            suffix = Path(f.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                doc = loader.load(tmp_path)
                if not doc.content:
                    st.warning(f"⚠️ {f.name} 无法解析")
                    continue
                chunks = splitter.split(doc.content, source=f.name)
                texts = [c.text for c in chunks]
                vecs = embedder.embed(texts).tolist()
                metas = [{"source": c.source, "chunk_index": c.index} for c in chunks]
                store.add(documents=texts, embeddings=vecs, metadatas=metas)
                st.session_state._doc_texts.extend(texts)
                st.session_state._doc_embeddings.extend(vecs)
                st.session_state._doc_metas.extend(metas)
                hybrid.index(
                    st.session_state._doc_texts,
                    st.session_state._doc_embeddings,
                    st.session_state._doc_metas,
                )
                st.success(f"✅ {f.name} → {len(chunks)} 块")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    st.divider()
    if st.button("📊 查看已入库文档"):
        n = store.count()
        if n == 0:
            st.info("暂无文档")
        else:
            _, metas, _ = store.search(embedder.embed_query("汇总").tolist(), top_k=n)
            seen = {}
            for m in metas:
                src = m.get("source", "未知")
                seen[src] = seen.get(src, 0) + 1
            for src, cnt in seen.items():
                st.write(f"📄 {src} ({cnt} 块)")

# --- 主体：多轮问答区 ---
st.header("💬 智能问答（多轮对话）")

col1, col2 = st.sidebar.columns(2)
with col1:
    top_k = st.sidebar.slider("检索块数", 1, 10, 3)
with col2:
    use_stream = st.sidebar.checkbox("流式输出", value=True)

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
question = st.chat_input("输入你的问题", key="chat_input")
if question:
    # 加用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 检索
    if st.session_state._doc_texts:
        coarse_k = min(top_k * 3, len(st.session_state._doc_texts))
        results = hybrid.hybrid_search(question, top_k=coarse_k)
        if results:
            cands = [r[0] for r in results]
            ranked = reranker.rerank(question, cands, top_k=top_k)
            docs = [r[0] for r in ranked]
            sources = []
            for d in docs:
                try:
                    idx = st.session_state._doc_texts.index(d)
                    sources.append(st.session_state._doc_metas[idx].get("source", ""))
                except ValueError:
                    sources.append("未知")
        else:
            docs, sources = [], []
    else:
        docs, sources = [], []

    # 生成（带历史）
    with st.chat_message("assistant"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]  # 不包括刚加的最新问题
        ]
        if use_stream:
            response_stream = generator.generate_stream_with_history(
                question, docs, sources, history=history,
            )
            answer = st.write_stream(response_stream)
        else:
            answer = generator.generate_with_history(
                question, docs, sources, history=history,
            )
            st.write(answer)

    # 保存回答
    st.session_state.messages.append({"role": "assistant", "content": answer})
