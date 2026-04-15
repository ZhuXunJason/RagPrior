# 基于 LLM 与 GraphPrior 的缺陷报告优先级排序

[English Documentation](README.md)

本项目提供两类主要排序模式：

- `llmprior`：直接用 LLM 做报告优先级排序
- `graphprior`：利用离线 KG、候选生成、选择式 LLM 链接、动态建簇、确定性打分和多样性重排的结构化排序

## 当前 GraphPrior 的真实实现

- 默认使用离线 KG，来源是 [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)
- 使用 KG 驱动的本地 mention 抽取与 KG 关系增强候选生成
- 在调用 LLM 之前先做确定性 prefill
- 只有歧义槽位才交给 LLM
- LLM 输出单行链接结果，例如 `R_001|O1|P0|E2|B3|page`
- GraphPrior 对 Qwen 默认关闭 thinking
- 对 `scope` 做确定性后校准

Neo4j 仍然可以通过 `--kg-mode online` 使用，但现在它是可选项，不再是默认依赖。

## GraphPrior 核心流程

### 阶段 1：候选生成与选择式链接

核心文件：[`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

对每条代表性报告，GraphPrior 会为 `Operation / Page / Element / Bug` 构造候选。候选来自三类证据：

- 基于 KG `surface / aliases / markers` 的当前报告直接 mention
- 离线 KG 文本匹配
- 运行时实例池提供的 fallback 候选

候选排序不是简单叠加分数，而是按证据层次排序：

1. 当前报告直接命中
2. offline KG 支持
3. operation 关系支持
4. memory fallback 支持

随后执行确定性 prefill：

- 没有候选：直接预填 `NONE` 或 `未知*`
- 只有一个强候选，或第一候选明显胜出：直接本地锁定
- 只有真正存在歧义的槽位才会进入 LLM

如果某条报告所有槽位都已经本地确定，则完全跳过 LLM。

进入 LLM 的报告只需要返回一行简短结果：

```text
R_001|O1|P0|E2|B3|page
```

这让解析更稳，也把 output token 压得很低。

### 阶段 2：实例对齐、签名构造与动态建簇

核心文件：[`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

完成链接后，每条报告会被转换成四维签名：

- `operation_eid`
- `page_eid`
- `element_eid`
- `bug_eid`

每个维度都与当前运行时维护的实例池做相似度匹配；如果没有足够接近的实例，就新建实例节点。

当两条报告的四维签名完全一致时，它们会进入同一个 cluster。

### 阶段 3：确定性打分与多样性重排

核心文件：[`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

当前 GraphPrior 保留可解释的确定性打分方式：

- `element`：影响组件数 `I = 1`
- `page`：`I = 该页面下的元素数`
- `program`：`I = 当前已知元素总数`

打分公式为：

```text
Score = I * log(T + 1)
```

其中 `T` 是该 cluster 的报告数。

之后会对 cluster 代表做带多样性惩罚的重排，尽量避免排序头部被近重复问题占满。

## 为什么仍然需要 GraphPrior

和纯 `llmprior` 相比，GraphPrior 更强调：

- 更低且更稳定的 output token
- 中间结构可见
- 候选来源可解释
- 抽取行为更可复查

这意味着它主动用一部分“结构化约束”来换取更低的波动和更高的可审计性。

## 评估指标

当前评估会输出：

- `APFD`：能否更早覆盖不同 bug 组
- `APSD`：带严重性权重的早发现能力
- `NDCG@10`、`NDCG@20`：排序头部质量

说明：

- 如果数据集没有外部严重性真值，当前 `APSD` 更接近“频次代理加权指标”，不是严格的真实 severity 指标
- 当前 GraphPrior 的主要优化目标是 `APFD / APSD + token 成本`，`NDCG` 作为辅助观察指标

## 环境要求

### 必需

- Python 环境和 `requirements.txt` 中的依赖
- 所选模型的 API Key
- 本地存在 [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)

### 可选

- Neo4j，仅当你需要 `--kg-mode online` 时才需要

## 安装

```bash
git clone https://github.com/ZhuXunJason/RAGPrior
cd RAGPrior
pip install -r requirements.txt
```

## 使用方法

### GraphPrior

默认命令：

```bash
python main.py --app 2 --model qwen3.5-plus --embedding-model text-embedding-v4
```

这里默认就是：

- `--mode graphprior`
- `--kg-mode offline`

如果想显式使用在线 KG：

```bash
python main.py --app 2 --model qwen3.5-plus --embedding-model text-embedding-v4 --kg-mode online
```

### LLMPrior

```bash
python main.py --app 2 --mode llmprior --model qwen3.5-plus
```

### 常用参数

- `--app <int>`：应用 id，对应 `reports/app{ID}.csv`
- `--file <path>`：自定义 CSV 文件
- `--model <str>`：LLM 名称
- `--embedding-model <str>`：embedding 模型
- `--kg-mode offline|online`：KG 后端，默认 `offline`
- `--batch-size <int>`：GraphPrior 抽取批大小
- `--workers <int>`：GraphPrior 并发线程数
- `--repeat <int>`：重复实验次数

## 输出结果

- [`response/`](/d:/OneDrive/Project/GraphPrior/GraphPrior/response)：原始 LLM 响应和 GraphPrior 调试信息
- [`result/`](/d:/OneDrive/Project/GraphPrior/GraphPrior/result)：最终排序结果

当前 GraphPrior 的 response 文件会额外包含：

- 每个 batch 的 token 统计
- 原始模型输出
- prefill 信息
- 每个槽位的 evidence 摘要
- 候选数与候选来源的调试信息

## 代码结构

- [`main.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/main.py)：主入口和评估流程
- [`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)：GraphPrior 主流程
- [`offline_kg.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/offline_kg.py)：离线 KG 解析与候选检索
- [`kg_service.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/kg_service.py)：在线 / 离线 KG 分发
- [`llm.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/llm.py)：模型调用层
- [`report_utils.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/report_utils.py)：报告读取与评估指标
- [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)：离线 KG 的实体与关系定义

## 当前局限

- 某些 app 上 `Page / Element` 的解析仍然偏保守
- 离线 KG 的效果依赖 [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md) 的覆盖度
- 当前打分具有可解释性，但还没有引入跨 app 学习型参数优化
- 除抽取阶段外，不同模型族在整体排序上仍可能有差异

## 许可证

[MIT License](LICENSE)
