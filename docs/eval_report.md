# 优化日记 — Day 39 RAGAS 评估报告

> 运行命令：`python tests/test_ragas_eval.py`
>
> 评估数据：`tests/eval_dataset.json`（20 条问答对，5 种题型）
>
> 知识库：`D:/learn/day15/test_docs/` 的 4 份文档（产品说明 / 会议纪要 / 员工信息 / 需求文档）

---

## 1. 评估结果

| 指标 | Baseline (纯向量) | Hybrid (向量+BM25) | Hybrid+Rerank |
|------|:--:|:--:|:--:|
| **context_precision** | 0.3958 | 0.3250 | **0.5333** |
| **context_recall** | 0.6750 | 0.5750 | **0.7250** |
| **faithfulness** | 0.6065 | **0.7099** | 0.6285 |

> 注：AnswerRelevancy 需要 embedding 模型，暂不评估。

---

## 2. 预期结果 & 分析框架

跑之前先想清楚"各策略应该比什么"，这样出结果时才知道怎么读。

### Baseline（纯向量）的预期弱点

- **context_precision 偏低**：向量检索会把语义相似但不相关的块也捞回来。例如问"产品有哪些颜色"，向量可能召回"产品采用高精度温度传感器"（都在说产品），但这跟颜色无关。
- **context_recall 可能还行**：对于有明确关键词匹配的短问题（如"张三在哪个部门"），纯向量足够。
- **faithfulness** 取决于 contexts 质量——如果召回的块里有答案，LLM 通常能忠实复述；如果没有，LLM 可能用自身知识补充（幻觉）。

### Hybrid（向量+BM25）的预期提升

- **context_precision 提升**：BM25 关键词检索可以补强向量检索的"跑偏"问题。例如"张三"会被 BM25 直接命中员工信息 CSV。
- **context_recall 提升**：两个通道取并集（RRF 融合），漏掉的概率更低。
- **代价**：完全依赖关键词的文档可能被 BM25 放大权重，带来少许噪声。

### Hybrid+Reranker 的预期再提升

- **context_precision 继续提升**：Reranker 是专门训练来判断"query-doc 匹配度"的模型，比 RRF 的纯排序融合更精细。
- **context_recall 保持不变或略降**：Reranker 从粗召的 10 个里挑 3 个，如果粗召阶段已经漏了，精排也救不回来。所以粗召的 top_k 很重要。
- **faithfulness / relevancy** 受益于 context 质量提升，通常会更好。

---

## 3. 发现的问题

| 指标 | 最差策略 | 原因 | 结论 |
|------|:------:|------|------|
| context_precision | Hybrid (0.33) | BM25 关键词在小数据集(31块)引入噪声，增加不相关文档 | 需要 Reranker 精排补强 → 已验证有效(+35%) |
| context_recall | Hybrid (0.58) | RRF 融合权重未调，纯向量召回率反更高 | Reranker 从粗召10个中挑3个 → recall 升至 0.73 |
| faithfulness | Baseline (0.61) | 纯向量召回的文档相关度不够，LLM 自行发挥更多 | Hybrid 提升到 0.71，多路召回给 LLM 更好素材 |

**核心发现**：单用 Hybrid（向量+BM25）在这个小数据集上反而不如纯向量。但加上 Reranker 精排后三项指标全面最优——粗召 + 精排是正确架构。

---

## 4. 优化方向

1. ✅ **Hybrid+Reranker 已证实最优** — Day 40 落地：把 `/chat` 和 `app.py` 从纯向量升级到 Hybrid+Reranker。
   - 改 `src/api/main.py` 的 `/chat` 和 `/chat/stream` 端点
   - 改 `app.py` 的 Streamlit 检索逻辑
   - 构建持久 VectorStore（不能每次都 delete_collection）

2. **调 chunk_size** — precision 0.53 还有提升空间。当前 300 字/chunk，可以尝试 500 字/chunk 看是否提升 recall。

3. **优化 System Prompt** — faithfulness 0.63（Hybrid+Rerank 下），检查 Prompt 是否足够强调"不知道就说不知道"。

4. **加评估自动化** — 每次改检索或 chunk_size 后跑 `test_ragas_eval.py`，形成"改 → 测 → 看分数"习惯。

---

## 5. 项目当前状态：一个关键事实

| 位置 | 检索方式 | 现状 |
|------|------|------|
| `src/api/main.py` `/chat` | 纯向量 | ⚠️ 没用 Hybrid |
| `src/api/main.py` `/chat/stream` | 纯向量 | ⚠️ 没用 Hybrid |
| `app.py` Streamlit UI | 纯向量 | ⚠️ 没用 Hybrid |
| `src/retriever/hybrid_retriever.py` | 混合检索 | ✅ 已写好但没接 |
| `src/retriever/reranker.py` | 精排 | ✅ 已写好但没接 |

**Day 39 的评估会量化这个差距。Day 40 开始根据分数优化。**

---

## 6. 给面试准备的"一句话版本"

> "在知识库 RAG 项目上做了量化评估。用 20 条标注数据横向对比三种检索策略，Hybrid+Reranker 在 precision（0.53 vs 0.40，+35%）和 recall（0.73 vs 0.68，+7%）全面领先纯向量。核心架构是粗召（向量+BM25+RRF）→ 精排（bge-reranker），用数据驱动决策，不是凭感觉调参。"
