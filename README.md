# 企业知识库 RAG 问答系统

基于 RAG 架构的企业内部知识库智能问答系统。

## 技术栈

- **LLM**: DeepSeek
- **Embedding**: bge-small-zh-v1.5
- **向量数据库**: Chroma
- **检索**: 向量检索 + BM25 多路召回 + Reranker 精排
- **后端**: FastAPI
- **文档解析**: PyMuPDF / python-docx

## 功能

- [x] 支持 PDF / DOCX / TXT / CSV 文档加载
- [x] 文档切块（递归降级 + overlap）
- [ ] Embedding + 向量库存储
- [ ] 语义检索 + BM25 多路召回
- [ ] Reranker 精排
- [ ] LLM 生成回答
- [ ] 多轮对话
- [ ] 答案引用来源标注
- [ ] REST API

## 开发进度

| 模块 | 天数 | 状态 |
|------|------|------|
| 项目骨架 + 依赖 | Day 29 | ✅ |
| 文档加载器 (loader/) | Day 30 | ✅ |
| 文本切分 (chunker/) | Day 31 | ✅ |

## 项目结构

\```
src/
├── loader/       # 文档加载
├── chunker/      # 文本切分
├── embedder/     # 向量化
├── retriever/    # 检索
├── generator/    # LLM 生成
└── api/          # FastAPI 接口
\```

## 快速开始

\```bash
poetry install
cp .env.example .env  # 编辑 .env 填入 DEEPSEEK_API_KEY
poetry run uvicorn src.api.main:app --reload
\```