"""核心数据结构（贯穿整个项目的统一格式）"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class Document(BaseModel):
    """从文件加载后的文档"""
    content: str                          # 正文
    source: str                           # 来源文件名
    file_type: str                        # pdf / docx / txt / csv
    page_count: int = 1


class Chunk(BaseModel):
    """切分后的文本块"""
    text: str                             # 块内容
    index: int                            # 块序号
    source: str                           # 来源文档名
    metadata: dict = {}                   # 附加元数据


class RetrievedChunk(BaseModel):
    """检索返回的文本块（带相似度分数）"""
    text: str
    score: float                          # 相似度分数
    source: str


class QueryResult(BaseModel):
    """一次 RAG 问答的完整结果"""
    question: str                         # 用户问题
    answer: str                           # LLM 生成的回答
    retrieved_chunks: List[RetrievedChunk]  # 检索到的文档块
    latency_ms: float                     # 响应耗时（毫秒）
