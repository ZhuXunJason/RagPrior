# 基于 LLM 与 GraphPrior 的缺陷报告优先级排序

本项目利用大语言模型（LLM）与知识图谱增强方法（GraphPrior），对众包测试报告进行优先级排序。GraphPrior 包含实例抽取融合、拓扑严重性打分、图向量混合去重与多样性排序四个阶段。

[English Documentation](README.md)

## 主要特性

*   **GraphPrior 四阶段流程**：实例抽取与融合、拓扑打分、混合去重、多样性排序。
*   **层次化聚类**：利用树状结构将缺陷报告细分为不同粒度的类型。
*   **多样性优先排序**：对报告进行排序，最大化前序报告中的缺陷种类覆盖率。
*   **自动评估**：统一计算 APSD 与 NDCG 指标，量化严重性发现效率与头部排序质量。

## 环境要求

1.  **Neo4j 数据库**：已运行并导入缺陷知识图谱数据的 Neo4j 实例。
2.  **API Keys**:
    *   **Google Gemini**: 访问 Gemini 模型。
    *   **OpenAI**: 访问 GPT 模型。

## 安装步骤

1.  克隆仓库：
    ```bash
    git clone https://github.com/ZhuXunJason/RAGPrior
    cd RAGPrior
    ```

2.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
 
## 使用方法

### 1. 配置环境
打开 `.env.example` 文件，填入实际配置信息并重命名为`.env`。

### 2. 数据准备
将缺陷报告 CSV 文件放置在 `reports/` 目录下（例如 `reports/app21.csv`）。

### 3. 运行排序
使用 `main.py` 运行排序流程。

**基本命令 (GraphPrior 模式):**
```bash
python main.py --app 21 --mode graphprior --repeat 1 --model gpt-5.2
```

**参数说明:**
*   `--app <int>`: 应用 ID（对应 `reports/app{ID}.csv`）。默认值: 21。
*   `--model <str>`: 模型名称（例如 `gemini-3-pro-preview`, `gpt-5.2`）。默认值: `gemini-3-pro-preview`。
*   `--mode <str>`: 运行模式。
    *   `graphprior`: GraphPrior 四阶段流程。
    *   `llmprior`: 标准 LLM 排序。
    *   `simple`: 简化版提示词。
    *   `direct`: 直接排序，不生成分类树。
*   `--repeat <int>`: 实验重复次数。默认值: 1。
*   `--file <path>`: 指定自定义 CSV 文件路径（覆盖 `--app` 参数）。

### 4. 输出结果
*   **Responses (`response/`)**: 保存 LLM 的原始文本响应。
*   **Results (`result/`)**: 保存解析后的最终报告排序序列。
*   **控制台输出**: 打印运行进度及最终 APSD / NDCG 评估分数。

## 代码结构

*   **`main.py`**: 程序入口。协调 查询 -> 解析 -> 评估 的全流程。
*   **`kg_service.py`**: 处理 Neo4j 连接、向量嵌入以及混合图谱检索逻辑。
*   **`llm.py`**: Google Gemini API 调用接口。
*   **`prompt.py`**: 管理提示词模板及上下文注入。
*   **`parse.py`**: 解析 LLM 输出的层次树结构，生成线性序列。
*   **`config.py`**: 配置加载器。

## 许可证

[MIT License](LICENSE)
