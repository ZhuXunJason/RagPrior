# 基于 LLM 和图谱增强检索 (Graph RAG) 的缺陷报告优先级排序

本项目利用大语言模型（LLM）结合知识图谱检索增强生成（Graph RAG）技术，对众包测试报告进行优先级排序。项目通过构建层次化的缺陷分类树，生成优化的报告阅读顺序，以确保审核人员能尽可能早地发现多样化的缺陷类型。

[English Documentation](README.md)

## 主要特性

*   **图谱增强检索 (Graph RAG)**：通过 Neo4j 知识图谱检索结构化上下文（操作步骤、UI组件），增强 LLM 对缺陷的理解。
*   **层次化聚类**：利用树状结构将缺陷报告细分为不同粒度的类型。
*   **多样性优先排序**：对报告进行排序，最大化前序报告中的缺陷种类覆盖率。
*   **自动评估**：计算 APFD (Average Percentage of Faults Detected) 指标，量化排序效果。

## 环境要求

1.  **Neo4j 数据库**：已运行并导入缺陷知识图谱数据的 Neo4j 实例。
2.  **Google Gemini API Key**：需要访问 Gemini 模型（如 `gemini-3-flash` 或 `gemini-embedding-001` 模型）的权限。

## 安装步骤

1.  克隆仓库：
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 配置指南

1.  打开 `.env.example` 文件，填入您的实际配置信息并重命名为.env。
 
## 使用方法

### 1. 配置环境
打开 `.env.example` 文件，填入实际配置信息并重命名为.env。

### 2. 数据准备
将您的缺陷报告 CSV 文件放置在 `reports/` 目录下（例如 `reports/app21.csv`）。

### 3. 运行排序
使用 `main.py` 运行排序流程。

**基本命令 (Graph RAG 模式):**
```bash
python main.py --app 21 --mode rag --repeat 1
```

**参数说明:**
*   `--app <int>`: 应用 ID（对应 `reports/app{ID}.csv`）。默认值: 21。
*   `--mode <str>`: 运行模式。
    *   `rag`: 使用知识图谱 RAG 增强（推荐）。
    *   `default`: 标准 LLM 排序，不使用 RAG。
    *   `simple`: 简化版提示词。
    *   `direct`: 直接排序，不生成分类树。
*   `--repeat <int>`: 实验重复次数。默认值: 1。
*   `--file <path>`: 指定自定义 CSV 文件路径（覆盖 `--app` 参数）。

### 4. 输出结果
*   **Responses (`response/`)**: 保存 LLM 的原始文本响应。
*   **Results (`result/`)**: 保存解析后的最终报告排序序列。
*   **控制台输出**: 打印运行进度及最终的 APFD 评估分数。

## 代码结构

*   **`main.py`**: 程序入口。协调 查询 -> 解析 -> 评估 的全流程。
*   **`kg_service.py`**: 处理 Neo4j 连接、向量嵌入以及混合图谱检索逻辑。
*   **`llm.py`**: Google Gemini API 调用接口。
*   **`prompt.py`**: 管理提示词模板及上下文注入。
*   **`parse.py`**: 解析 LLM 输出的层次树结构，生成线性序列。
*   **`config.py`**: 配置加载器。

## 许可证

[MIT License](LICENSE)
