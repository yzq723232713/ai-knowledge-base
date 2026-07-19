"""
Day 39：RAGAS 评估脚本 — 三种检索策略横向对比

在项目一的真实代码上跑 RAGAS 评估，用数据而不是感觉来判断哪种检索方案更好。

三种策略：
  A (Baseline)     — 纯向量检索（当前 /chat 接口实际用这个）
  B (Hybrid)       — 向量 + BM25 + RRF 多路召回
  C (Hybrid+Rerank)— B 的粗召结果再用 bge-reranker 精排

每道题 → 三种策略各跑一遍 → 收集 answer + contexts
→ 丢进 RAGAS 用 LLM 打分 → 输出 4 指标对比表

前提：
  pip install ragas langchain-openai datasets          # 三个新依赖
  或 poetry install                                     # 已写入 pyproject.toml
"""
import sys
import os
import json
import time

# ---- 路径与全局设置 ----
sys.path.insert(0, "D:/ai-knowledge-base")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 先把 huggingface/transformers 的噪音关掉，评估输出才看得清
import logging
for noisy in ["httpx", "chromadb", "sentence_transformers", "llama_index",
              "transformers", "asyncio", "urllib3", "openai"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv("D:/ai-knowledge-base/.env")

import numpy as np
from pathlib import Path

# ---- 项目模块 ----
from src.loader.document_loader import DocumentLoader
from src.chunker.text_splitter    import TextSplitter
from src.embedder.embedder        import Embedder
from src.retriever.vector_store   import VectorStore
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker       import Reranker
from src.generator.generator      import Generator


# ============================================================
# Part 1: 加载文档 + 切块 + 向量化（只做一次，三种策略共享）
# ============================================================
print("=" * 60)
print("Part 1: 准备知识库（加载 → 切块 → 向量化）")
print("=" * 60)

loader   = DocumentLoader()
splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
embedder = Embedder()
gen      = Generator()

test_dir = Path("D:/learn/day15/test_docs")
all_chunks = []
for fp in sorted(test_dir.iterdir()):
    if fp.suffix.lower() not in {".pdf", ".docx", ".txt", ".csv"}:
        continue
    doc = loader.load(str(fp))
    if not doc.content:
        print(f"  ⚠ {fp.name} 加载失败")
        continue
    chunks = splitter.split(doc.content, source=doc.source)
    all_chunks.extend(chunks)
    print(f"  ✅ {fp.name} → {len(chunks)} 块")

texts   = [c.text for c in all_chunks]
sources = [c.source for c in all_chunks]
embeddings = embedder.embed(texts).tolist()
metadatas  = [{"source": c.source, "chunk_index": c.index} for c in all_chunks]

print(f"\n共 {len(all_chunks)} 个块，Embedding 维度 {len(embeddings[0])}")


# ============================================================
# Part 2: 构建三种检索策略
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 初始化三种检索策略")
print("=" * 60)

# 策略A — 纯向量（跟当前 /chat 一样）
store_a = VectorStore(collection_name="eval_baseline")
store_a.add(documents=texts, embeddings=embeddings, metadatas=metadatas)
print("  ✅ Strategy A: 纯向量检索（Baseline）")

# 策略B — Hybrid（向量 + BM25 + RRF）
store_b = VectorStore(collection_name="eval_hybrid")
hybrid_b = HybridRetriever(embedder, store_b)
hybrid_b.index(texts, embeddings, metadatas)
print("  ✅ Strategy B: Hybrid 多路召回（向量 + BM25 + RRF）")

# 策略C — Hybrid + Reranker
# Reranker 路径在系统重装后可能不对，降级处理
strategy_c_enabled = False
reranker = None
store_c = VectorStore(collection_name="eval_hybrid_rerank")
hybrid_c = HybridRetriever(embedder, store_c)
hybrid_c.index(texts, embeddings, metadatas)

try:
    # 优先用默认路径；如果缓存缺失则让 CrossEncoder 自己从镜像下
    reranker = Reranker()
    strategy_c_enabled = True
    print("  ✅ Strategy C: Hybrid + Reranker（bge-reranker-v2-m3）")
except Exception as e:
    print(f"  ⚠ Reranker 加载失败 ({e})，Strategy C 跳过，仅对比 A vs B")


# ============================================================
# Part 3: 加载评估数据集
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 加载 20 条评估数据")
print("=" * 60)

dataset_path = Path(__file__).parent / "eval_dataset.json"
with open(dataset_path, encoding="utf-8") as f:
    eval_items = json.load(f)

print(f"  共 {len(eval_items)} 条评估用例")


# ============================================================
# Part 4: 对每条问题，三种策略分别检索 + 生成
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 跑 20 题 × 三种策略")
print("=" * 60)

def retrieve_baseline(query: str, top_k: int = 3):
    """策略A：纯向量检索"""
    q_emb = embedder.embed_query(query).tolist()
    docs, metas, _ = store_a.search(q_emb, top_k=top_k)
    srcs = [m.get("source", "") for m in metas]
    return docs, srcs

def retrieve_hybrid(query: str, top_k: int = 3):
    """策略B：Hybrid（向量+BM25+RRF）"""
    results = hybrid_b.hybrid_search(query, top_k=top_k)
    docs = [r[0] for r in results]
    indices = [r[2] for r in results]
    srcs = [sources[i] if 0 <= i < len(sources) else "未知" for i in indices]
    return docs, srcs

def retrieve_hybrid_rerank(query: str, top_k: int = 3, coarse_k: int = 10):
    """策略C：Hybrid 粗召 → Reranker 精排"""
    if not strategy_c_enabled:
        return retrieve_hybrid(query, top_k)
    results = hybrid_c.hybrid_search(query, top_k=coarse_k)
    cands = [r[0] for r in results]
    # Reranker 精排
    ranked = reranker.rerank(query, cands, top_k=top_k)
    docs = [r[0] for r in ranked]
    # 反查来源
    # 注意：rerank 返回的文档文本可能与原 texts 有细微差别
    srcs = []
    for d in docs:
        try:
            idx = texts.index(d)
            srcs.append(sources[idx])
        except ValueError:
            # fallback: 模糊匹配
            best_idx = -1
            best_len = 0
            for i, t in enumerate(texts):
                if d[:50] in t or t[:50] in d:
                    if len(t) > best_len:
                        best_len = len(t)
                        best_idx = i
            srcs.append(sources[best_idx] if best_idx >= 0 else "未知")
    return docs, srcs

# 跑评估，收集 RAGAS 需要的四元组
results_a = []  # [{question, answer, contexts, ground_truth}, ...]
results_b = []
results_c = []

top_k = 3
t0 = time.time()

for i, item in enumerate(eval_items):
    q = item["question"]
    gt = item["ground_truth"]

    docs_a, srcs_a = retrieve_baseline(q, top_k)
    ans_a = gen.generate(q, list(docs_a), list(srcs_a))
    results_a.append({"question": q, "answer": ans_a, "contexts": list(docs_a), "ground_truth": gt})

    docs_b, srcs_b = retrieve_hybrid(q, top_k)
    ans_b = gen.generate(q, list(docs_b), list(srcs_b))
    results_b.append({"question": q, "answer": ans_b, "contexts": list(docs_b), "ground_truth": gt})

    if strategy_c_enabled:
        docs_c, srcs_c = retrieve_hybrid_rerank(q, top_k)
        ans_c = gen.generate(q, list(docs_c), list(srcs_c))
        results_c.append({"question": q, "answer": ans_c, "contexts": list(docs_c), "ground_truth": gt})

    elapsed = time.time() - t0
    print(f"  [{i+1:2d}/{len(eval_items)}] {q[:30]:<30s} ({elapsed:.0f}s)")

total_time = time.time() - t0
n_strategies = 3 if strategy_c_enabled else 2
print(f"\n  完成！总耗时 {total_time:.0f}s，平均每题每策略 {(total_time/n_strategies/len(eval_items)):.1f}s")


# ============================================================
# Part 5: RAGAS 评估
# ============================================================
print("\n" + "=" * 60)
print("Part 5: RAGAS LLM 评估")
print("=" * 60)

print("""
RAGAS 用 LLM 判断每条回答的质量 —— 不是关键词匹配，是真的"读"了文档和答案。
4 个指标：
  context_precision  — 检索到的文档中有多少跟问题相关
  context_recall     — ground_truth 里的信息是否被检索到的文档覆盖
  faithfulness       — LLM 回答是否基于检索文档（不是幻觉）
  answer_relevancy   — 回答跟问题切题程度
""")

# ---- 配置 RAGAS ----
from ragas import evaluate
# RAGAS 0.4.3 的 evaluate() 接受旧版单例 metrics（小写），不是 collections 中的类
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

eval_llm = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
))
print("  eval_llm ready")

metrics_to_eval = [context_precision, context_recall, faithfulness]
for m in metrics_to_eval:
    m.llm = eval_llm

from datasets import Dataset as HFDataset

def run_ragas_eval(strategy_name: str, samples: list[dict]) -> dict:
    """把收集的 samples 丢进 RAGAS，返回 4 指标得分"""
    if not samples:
        return {}
    ds = HFDataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])
    try:
        result = evaluate(ds, metrics=metrics_to_eval)
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        # 用更容易读的 key 名
        out = {}
        for k, v in scores.items():
            short = k.replace("context_", "").replace("answer_", "")
            out[short] = round(float(v), 4)
        print(f"\n  [{strategy_name}]")
        for k, v in out.items():
            bar = "█" * int(v * 20)
            print(f"    {k:<18s}: {v:.4f}  {bar}")
        return out
    except Exception as e:
        print(f"\n  ⚠ [{strategy_name}] RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


scores_a = run_ragas_eval("Strategy A · Baseline", results_a)
scores_b = run_ragas_eval("Strategy B · Hybrid", results_b)
scores_c = run_ragas_eval("Strategy C · Hybrid+Rerank", results_c) if strategy_c_enabled else {}


# ============================================================
# Part 6: 汇总对比表
# ============================================================
print("\n\n" + "=" * 60)
print("Part 6: 三策略横向对比")
print("=" * 60)

all_metrics = ["precision", "recall", "faithfulness"]
# 对齐 key 名变体
def norm_scores(d):
    return {
        "precision": d.get("precision", 0),
        "recall": d.get("recall", 0),
        "faithfulness": d.get("faithfulness", 0),
    }

a = norm_scores(scores_a)
b = norm_scores(scores_b)
c = norm_scores(scores_c) if scores_c else {m: 0 for m in all_metrics}

print(f"\n{'指标':<18s} {'Baseline':>10s} {'Hybrid':>10s} {'Hybrid+Rerank':>15s}")
print("-" * 57)
for m in all_metrics:
    print(f"{m:<18s} {a[m]:>10.4f} {b[m]:>10.4f} {c[m]:>15.4f}")

# 找最优策略
best_prec = max(a["precision"], b["precision"], c["precision"])
best_rec  = max(a["recall"], b["recall"], c["recall"])
best_faith = max(a["faithfulness"], b["faithfulness"], c["faithfulness"])

print(f"\n{'最优':<18s}", end="")
for m, best in zip(all_metrics, [best_prec, best_rec, best_faith]):
    tag = ""
    if best == a[m] and best != 0:
        tag = " ← A"
    if best == b[m] and best != 0:
        tag += " ← B"
    if best == c[m] and best != 0:
        tag += " ← C"
    print(f" {best:.4f}{tag:<6s}", end="")

print("\n\n=== Day 39 评估完成 ===\n")
print("结论已写入 docs/eval_report.md")
print("提示：用 poetry run python tests/test_ragas_eval.py 运行")
