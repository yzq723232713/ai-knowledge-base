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
from src.generator.generator import Generator

# ===== 初始化 =====
@st.cache_resource
def init_components():
    return {
        "loader": DocumentLoader(),
        "splitter": TextSplitter(chunk_size=300, chunk_overlap=50),
        "embedder": Embedder(),
        "store": VectorStore(collection_name="streamlit_kb"),
        "generator": Generator(),
    }

comp = init_components()
loader = comp["loader"]
splitter = comp["splitter"]
embedder = comp["embedder"]
store = comp["store"]
generator = comp["generator"]

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

# --- 主体：问答区 ---
st.header("💬 智能问答")
question = st.text_input("输入你的问题", placeholder="例如：设备怎么安装？")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("检索块数", 1, 10, 3)
with col2:
    use_stream = st.checkbox("流式输出", value=True)

if st.button("🔍 提问") and question:
    # 检索
    q_emb = embedder.embed_query(question).tolist()
    docs, metas, _ = store.search(q_emb, top_k=top_k)
    sources = [m.get("source", "") for m in metas]

    # 显示检索结果
    with st.expander("📋 检索到的参考资料", expanded=False):
        for i, (d, s) in enumerate(zip(docs, sources)):
            st.markdown(f"**参考{i+1} · {s}**")
            st.text(d[:200] + ("..." if len(d) > 200 else ""))

    # 生成回答
    with st.chat_message("assistant"):
        if use_stream:
            response_stream = generator.generate_stream(question, docs, sources)
            st.write_stream(response_stream)
        else:
            answer = generator.generate(question, docs, sources)
            st.write(answer)
