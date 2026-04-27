# LLMPrior

基于 LLM 和 KG-RAG 的众包测试报告优先级排序工具。

## 环境准备

1. 安装依赖：`pip install -r requirements.txt`

2. 在项目根目录创建 `.env` 文件：

```
OPENAI_API_KEY=your_openai_key
BASE_URL=https://api.openai.com/v1
GOOGLE_API_KEY=your_google_key
EMBEDDING_MODEL=gemini-embedding-2-preview
DUPLICATE_THRESHOLD=0.85
```

- `BASE_URL` 可选，可指向自定义的 OpenAI 兼容端点。
- `GOOGLE_API_KEY` 在 KG 模式下用于 embedding 预聚类。
- `EMBEDDING_MODEL` 和 `DUPLICATE_THRESHOLD` 是 KG 模式的可选配置。

## 使用方式

将 `app{n}.csv` 文件放在 `./reports/` 目录下（在 `prompt.py` 中设置 `DATASET_ROOT`）。

### main.py — 纯 LLM 基线

```bash
python main.py --app 1 --repeat 3 --mode default --model gpt-5.4
```

模式：`default`（LLMPrior）、`simple`（SimpleLLMPrior）、`direct`（DirectLLMPrior）。

### main_kg.py — KG-RAG 流水线

```bash
# 完整 KG 流水线（Phase 1-4）
python main_kg.py --app 7 --repeat 3 --mode kg \
    --model deepseek-v4-flash \
    --dedup-threshold 0.85 \
    --skip-severity

# KG-only 贪心排序基线（无 LLM 排序调用）
python main_kg.py --app 7 --repeat 3 --mode kg-only --no-skip-severity
```

### 网格实验

```powershell
# run_exp.ps1：遍历 app 1-8，两个 dedup 阈值
.\run_exp.ps1

# 然后评估并生成对比表：
python eval_table.py
```

## KG-RAG 流水线详解（main_kg.py）

KG-RAG 增强流水线包含以下四个阶段：

### Phase 0 — Embedding 预聚类（去重）
- 使用配置的 embedding 模型（默认 `gemini-embedding-2-preview`）对所有报告描述进行向量化。
- 根据余弦相似度与 `DUPLICATE_THRESHOLD` 阈值将相似报告聚为一类。
- 每类选一个代表报告，减少后续阶段的处理量。
- LLM 排序完成后，通过 `expand_ranked_sequence` 将压缩的排序序列展开回所有原始报告。

### Phase 1 — 实体提取
- 对每份报告，使用 LLM 提取结构化实体：操作步骤、缺陷症状、涉及页面/元素、页面类型、功能模块。
- 结果缓存在 `./cache/app{n}_entities.json`，避免重复调用。

### Phase 2 — 知识图谱构建与分类
- 从提取的实体构建 `BugKnowledgeGraph`——节点代表页面、操作、症状和模块，边代表共现关系。
- 使用图匹配（TF-IDF 加权）将每份报告分类到具体的 bug 类型。
- 匹配置信度低的报告回退到 LLM 分类。

### Phase 3 — 严重性推导
- 基于知识图谱的中心性和频率计算每种 bug 类型的严重性分数。
- 如果设置 `--skip-severity`，所有报告使用中性严重性（priority=0.0）。
- `kg` 模式在 LLM 排序 prompt 中包含 severity 信息；`kg-only` 模式按 severity 降序贪心排序。

### Phase 4 — 报告排序

**`kg` 模式（默认）：** 将代表报告 + KG 摘要拼接成结构化 prompt，调用 LLM 生成优先级排序序列，然后展开回所有报告。

**`kg-only` 模式：** 贪心排序——从每个 bug 类型中选出 severity 最高的报告优先排列（保证多样性），其余报告按 severity 降序追加。零额外 LLM 成本。

## 代码结构

| 文件 | 作用 |
|------|------|
| `main.py` | 纯 LLM 流水线入口（query、parse、evaluate） |
| `main_kg.py` | KG-RAG 流水线入口（run_pipeline、run_evaluate） |
| `llm.py` | OpenAI 兼容 LLM 调用接口 |
| `prompt.py` | Prompt 模板和报告加载 |
| `prompt_kg.py` | KG 专用的 prompt 模板 |
| `parse.py` | LLM 响应解析和排序序列提取 |
| `config.py` | 环境配置和客户端工厂 |
| `kg/extractor.py` | Phase 1：实体提取 |
| `kg/graph.py` | BugKnowledgeGraph 数据结构 |
| `kg/classifier.py` | Phase 2：基于图的 bug 分类 |
| `kg/severity.py` | Phase 3：严重性分数计算 |
| `kg/dedup.py` | Phase 0：基于 embedding 的预聚类 |
| `eval_table.py` | 评估结果并打印对比表 |
| `run_exp.ps1` | 网格实验的 PowerShell 脚本 |

## 注意事项

- 如果 `run_parse` 失败，可手动修正 LLM 响应格式，或使用 `given_range` 重新运行特定轮次。
- 如果 `run_evaluate` 失败，说明生成的排序序列无效——需要重新运行。
- 使用 `--force-refresh` 可强制重新提取实体（忽略缓存）。
