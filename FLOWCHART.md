# GraphPrior 项目流程图 / Project Flowchart

## 1. 整体系统架构与数据流 / Overall System Architecture & Data Flow

```mermaid
flowchart TD
    A([📄 输入: Bug报告 CSV\nreports/appN.csv\n字段: index, description, img_url, bug_id]) --> B

    B{执行模式选择\n--mode}

    B -- graphprior --> C[GraphPrior 四阶段流水线\ngraph_prior.py]
    B -- llmprior --> D[LLM树形分类\nllm.py + parse.py]
    B -- direct --> E[LLM直接分类+排序\nllm.py + parse.py]
    B -- simple --> F[简化LLM提示\nllm.py + parse.py]
    B -- embedding --> G[嵌入相似度去重\nembedding.py]

    C --> H
    D --> H
    E --> H
    F --> H
    G --> H

    H([📊 输出: 优先级序列\nresult/appN-mode-iter-model.txt\n格式: Report 1 \[Score: 2.5\]\nReport 5 \[Score: 1.8\]\n...]) --> I

    I[评估阶段\nreport_utils.py]
    I --> J([📈 评估指标\nAPSD: 平均严重性分布百分比\nNDCG@10 / NDCG@20: 排序质量])

    style A fill:#d4edda,stroke:#28a745
    style H fill:#d1ecf1,stroke:#17a2b8
    style J fill:#fff3cd,stroke:#ffc107
    style C fill:#e2d9f3,stroke:#6f42c1
```

---

## 2. GraphPrior 四阶段核心流水线 / GraphPrior 4-Phase Core Pipeline

```mermaid
flowchart TD
    IN([输入报告列表\nList of Bug Reports]) --> P1

    subgraph P1["阶段1: 结构提取 & 实例融合 (Phase 1: Extraction & Instance Fusion)"]
        direction TB
        P1A[分批次处理\nbatch_size=2, workers=4] --> P1B
        P1B["LLM提取结构化属性\n(llm.py + kg_service.py)\n- Instance_Page: 页面描述\n- Instance_Element: 组件描述\n- Instance_Bug: 缺陷现象\n- Concept_BugType: 缺陷类型\n- Concept_BugSymptom: 症状\n- Concept_Operation: 操作\n- Concept_Severity: 严重等级"] --> P1C
        P1C["生成向量嵌入\n(embedding.py)\n- 页面嵌入 page_emb\n- 组件嵌入 elem_emb\n- 缺陷嵌入 bug_emb\n- 全文嵌入 doc_emb"] --> P1D
        P1D["实例对齐 & 融合\n余弦相似度 > 0.9\n相同类型实例合并为同一节点"] --> P1E
        P1E["构建内存图\n(networkx MultiDiGraph)\n节点: Report / Instance / Concept\n边: HAS_PAGE / HAS_ELEMENT / HAS_BUG / MAPPED_TO"]
    end

    P1E --> P2

    subgraph P2["阶段2: 拓扑严重性评分 (Phase 2: Topology-Based Severity Scoring)"]
        direction TB
        P2A["分析图结构拓扑\n- 识别页面级故障关键词\n  白屏/闪退/卡死/崩溃/无响应"] --> P2B
        P2B["计算节点度数\n- 统计每页面上的组件度数\n- degree = 受影响组件的连接数"] --> P2C
        P2C["计算严重性分数\nscore = max(1.0, element_degree)\n页面级故障额外加权"]
    end

    P2C --> P2OUT([score_map\n报告编号 → 严重性分数])

    P2OUT --> P3

    subgraph P3["阶段3: 混合图/向量去重 (Phase 3: Hybrid Graph/Vector De-duplication)"]
        direction TB
        P3A["按分数降序排列报告"] --> P3B
        P3B{"贪心聚类\n对每个未分配报告"}
        P3B --> P3C["硬匹配 (Hard Match)\nelement_eid 相同\nAND bug_eid 相同\n→ 判定为重复"]
        P3B --> P3D["软匹配 (Soft Match)\ncosine_sim(doc_emb) > 0.88\nAND Concept_Operation 相同\n→ 判定为重复"]
        P3C --> P3E["分配到现有簇\n保留高分报告为代表"]
        P3D --> P3E
        P3B --> P3F["无匹配 → 创建新簇\n该报告成为代表"]
        P3E --> P3G["输出:\nround1 = 代表报告列表\nround2 = 重复报告列表"]
        P3F --> P3G
    end

    P3G --> OUT

    OUT(["最终优先级序列\nsequence = round1 + round2\n[高严重性&独特报告优先]"])

    style IN fill:#d4edda,stroke:#28a745
    style OUT fill:#d1ecf1,stroke:#17a2b8
    style P1 fill:#f8f0ff,stroke:#6f42c1
    style P2 fill:#fff8e1,stroke:#ff8f00
    style P3 fill:#e8f5e9,stroke:#2e7d32
```

---

## 3. 知识图谱辅助检索流程 / Knowledge Graph Assisted Retrieval Flow

```mermaid
flowchart LR
    QT([查询文本\nQuery Text]) --> KGS

    subgraph KGS["KGService (kg_service.py)"]
        direction TB
        KGS1["extract_entities()\n提取操作词 & 组件词"] --> KGS2
        KGS2["expand_entities()\n通过 similar_to 关系\n在KG中扩展实体"] --> KGS3
        KGS3["向量相似度检索\ncosine_sim(query_emb, node_emb)"] --> KGS4
        KGS4["图结构上下文构建\n- 关联BugType\n- 关联BugSymptom\n- 关联Operation / Element"]
    end

    KGS4 --> CTX([检索上下文\nRAG Context])

    subgraph NEO4J["Neo4j 知识图谱"]
        direction TB
        N1["BugType 节点\n(10种: 功能缺陷/崩溃问题/布局问题/...)"]
        N2["BugSymptom 节点\n(33种: 未调用/不生效/崩溃/白屏/...)"]
        N3["Operation 节点\n(34种: 点击/提交/输入/刷新/...)"]
        N4["Element 节点\n(26种: 按钮/输入框/列表/弹窗/...)"]
        N5["Page 节点\n(5种: 用户交互类/数据展示类/...)"]
        N6["Module 节点\n(6种: 用户账号/订单交易/...)"]
        N1 -- associatedWith --> N2
        N1 -- TriggeredBy --> N3
        N1 -- Involves --> N4
        N5 -- belongsTo --> N6
        N1 -- similar_to --> N1
    end

    NEO4J <-->|Cypher查询| KGS3

    CTX --> LLM["LLM提示增强\n(RAG)\nllm.py → ask_gpt()"]

    style QT fill:#d4edda,stroke:#28a745
    style CTX fill:#d1ecf1,stroke:#17a2b8
    style LLM fill:#fce4ec,stroke:#e91e63
    style NEO4J fill:#e3f2fd,stroke:#1976d2
    style KGS fill:#f3e5f5,stroke:#7b1fa2
```

---

## 4. 模块依赖关系 / Module Dependency Graph

```mermaid
flowchart TD
    CLI["main.py\nCLI 入口 & 流程编排"]

    CLI --> GP["graph_prior.py\nGraphPrior 四阶段流水线"]
    CLI --> EMB["embedding.py\n嵌入生成 & 去重"]
    CLI --> PRS["parse.py\nLLM输出树形解析"]
    CLI --> RU["report_utils.py\nAPSD / NDCG 评估指标"]
    CLI --> PRO["prompt.py\n提示模板 & 报告加载"]
    CLI --> LLM["llm.py\nLLM API 接口\n(Gemini / OpenAI)"]

    GP --> LLM
    GP --> EMB
    GP --> KGS["kg_service.py\nNeo4j 知识图谱服务"]

    KGS --> LLM
    KGS --> DB[("Neo4j\n图数据库")]

    PRO --> KGS

    CFG["config.py\n配置 & 环境变量"]
    CFG --> CLI
    CFG --> KGS
    CFG --> LLM

    style CLI fill:#e2d9f3,stroke:#6f42c1,font-weight:bold
    style GP fill:#f8f0ff,stroke:#6f42c1
    style KGS fill:#e3f2fd,stroke:#1976d2
    style DB fill:#bbdefb,stroke:#1565c0
    style LLM fill:#fce4ec,stroke:#e91e63
    style EMB fill:#e8f5e9,stroke:#2e7d32
    style PRS fill:#fff8e1,stroke:#f57f17
    style RU fill:#fff3e0,stroke:#e65100
    style CFG fill:#f5f5f5,stroke:#616161
```

---

## 5. LLMPrior / Simple / Direct 模式流程 / Alternative Mode Flows

```mermaid
flowchart TD
    IN([输入报告]) --> PRO

    PRO["prompt.py\nload_reports_str()\n格式化报告文本\n可选: KG检索上下文注入"]

    PRO --> LLM["llm.py\nask_gpt()\n发送提示到 LLM\n获取树形/列表输出"]

    LLM -- llmprior/direct --> RESP["response/\nappN-mode-iter-model.txt\n原始LLM响应"]

    RESP --> PARSE["parse.py\nTreeNode 层级树解析\nselect_node() 遍历\ngenerate_sequence() 生成线性序列"]

    PARSE --> RES["result/\nappN-mode-iter-model.txt\n优先级排序结果"]

    RES --> EVAL["report_utils.py\ncompute_apsd_from_sequence()\ncompute_ndcg_from_sequence()"]

    EVAL --> METRICS([APSD / NDCG 分数\n打印到控制台])

    style IN fill:#d4edda,stroke:#28a745
    style METRICS fill:#fff3cd,stroke:#ffc107
    style RESP fill:#f8d7da,stroke:#dc3545
    style RES fill:#d1ecf1,stroke:#17a2b8
```

---

## 6. 数据结构流转 / Data Structure Flow

```mermaid
flowchart LR
    subgraph INPUT["输入数据结构"]
        direction TB
        CSV["CSV文件\nindex: str\ndescription: str\nimg_url: str\nbug_id: int"]
    end

    subgraph PROC["处理中间结构"]
        direction TB
        REPORT["Report Dict\n{\n  index: str\n  description: str\n  img_url: str\n  bug_id: int\n  severity_score: float\n}"]
        EXROW["Extracted Row\n{\n  report_num: str\n  description: str\n  Instance_Page: str\n  Instance_Element: str\n  Instance_Bug: str\n  Concept_BugType: str\n  Concept_BugSymptom: str\n  Concept_Operation: str\n  Concept_Severity: str\n  page_emb: np.array\n  elem_emb: np.array\n  bug_emb: np.array\n  doc_emb: np.array\n}"]
        GRAPH["NetworkX MultiDiGraph\n节点: Report/Instance/Concept\n边: HAS_/MAPPED_TO"]
        SCORE["score_map\n{ report_num: float }"]
    end

    subgraph OUTPUT["输出数据结构"]
        direction TB
        SEQ["优先级序列\n[report_num_1, report_num_2, ...]\n按 严重性+独特性 排序"]
        RESULT["结果文件\nReport 1 \[Score: 2.5\]:\nReport 5 \[Score: 1.8\]:\n..."]
    end

    CSV -->|load_reports_from_file| REPORT
    REPORT -->|Phase 1: LLM提取| EXROW
    EXROW -->|Phase 1: 实例融合| GRAPH
    GRAPH -->|Phase 2: 拓扑分析| SCORE
    SCORE -->|Phase 3: 去重排序| SEQ
    SEQ -->|写入文件| RESULT

    style INPUT fill:#d4edda,stroke:#28a745
    style PROC fill:#e8eaf6,stroke:#3f51b5
    style OUTPUT fill:#d1ecf1,stroke:#17a2b8
```
