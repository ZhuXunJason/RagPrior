# LLM-Based Bug Report Prioritization with GraphPrior

This project utilizes Large Language Models (LLM) with GraphPrior to prioritize crowdsourced test reports. GraphPrior includes four stages: extraction & instance fusion, topology-based severity scoring, hybrid graph/vector de-duplication, and diversity-first ranking.

[中文文档](README_CN.md)

## Features

*   **GraphPrior 4-Phase Pipeline**: extraction/fusion, topology scoring, hybrid de-duplication, and diversity ranking.
*   **Hierarchical Clustering**: Categorizes bugs into fine-grained types using a tree structure.
*   **Diversity-Based Prioritization**: Sorts reports to maximize the variety of bugs reviewed in top positions.
*   **Automated Evaluation**: Calculates APSD and NDCG to measure prioritization effectiveness.

## Detailed GraphPrior 4-Phase Workflow

GraphPrior separates end-to-end prioritization into four composable phases: report understanding, severity estimation, duplicate suppression, and final ranking.

### Phase 1: Extraction & Instance Fusion

*   **Input**: Raw bug report text (title, description, reproduction steps, device/version context).
*   **Process**: Use LLMs to extract bug-related instances (module, abnormal behavior, trigger conditions, impact scope), then align and fuse them with existing knowledge graph entities.
*   **Output**: A normalized instance set and report-to-instance mappings.
*   **Why it matters**: Converts unstructured reports into graph-usable semantic structures for downstream scoring and deduplication.

### Phase 2: Topology-based Severity Scoring

*   **Input**: Instance subgraphs and entity relations from Phase 1.
*   **Process**: Estimate potential severity using graph topology signals (e.g., relation density, key-node connectivity, impact propagation patterns).
*   **Output**: Initial severity scores for each report.
*   **Why it matters**: Provides a structure-aware severity prior instead of relying only on textual wording.

### Phase 3: Hybrid Graph/Vector De-duplication

*   **Input**: Text embeddings and graph-structure representations of reports.
*   **Process**: Combine semantic similarity (vector space) and structural similarity (graph space) to detect duplicate candidates and cluster them.
*   **Output**: De-duplicated report clusters and representative reports.
*   **Why it matters**: Reduces ranking distortion caused by repeated reports.

### Phase 4: Diversity-first Ranking

*   **Input**: Severity scores from Phase 2 and deduplication clusters from Phase 3.
*   **Process**: Rank reports by balancing severity priority with category coverage/information gain so top positions contain both critical and diverse issues.
*   **Output**: Final prioritized report list.
*   **Why it matters**: Improves early-review efficiency under limited triage budget.

### End-to-end Summary

1. Structure raw reports into graph-aligned instances (Phase 1).
2. Estimate severity with topology-aware signals (Phase 2).
3. Merge/suppress duplicates with hybrid similarity (Phase 3).
4. Produce a severity-and-diversity balanced ranking (Phase 4).

## Current Issues and Known Limitations

*   **Scoring still depends on specific LLMs**: Severity scoring and final ranking are still strongly coupled to the reasoning behavior of the selected model, so outputs can vary across model families and model versions instead of relying on a fully stable model-agnostic algorithm.
*   **Duplicate detection is not yet precise enough**: The hybrid graph/vector de-duplication can still misclassify semantically rewritten, cross-module, or long-form reports (false merges or missed duplicates).
*   **Token consumption is high**: The 4-phase pipeline (extraction, fusion, de-duplication, ranking) can require substantial context tokens and API calls on medium/large report sets, increasing latency and cost.
*   **Reproducibility is sensitive to model updates**: Even with fixed prompts and parameters, online model updates may shift results, so experiments need explicit model-version and timestamp tracking.

## Prerequisites

1.  **Neo4j Database**: A running Neo4j instance with the bug knowledge graph populated.
2.  **API Keys**: 
    *   **Google Gemini**: Required for Gemini models.
    *   **OpenAI**: Required for GPT models.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ZhuXunJason/RAGPrior
    cd RAGPrior
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Environment Configeration
Open `.env.example` and fill in your actual credentials. Rename it to `.env`.

### 2. Data Preparation
Place your bug report CSV files in the `reports/` directory (e.g., `reports/app21.csv`).

### 3. Run Prioritization
Use `main.py` to run the prioritization process.

**Basic Command (GraphPrior Mode):**
```bash
python main.py --app 21 --mode graphprior --repeat 1 --model gpt-5.2
```

**Parameters:**
*   `--app <int>`: The App ID (corresponding to `reports/app{ID}.csv`). Default: 21.
*   `--model <str>`: Model name (e.g., `gemini-3-pro-preview`, `gpt-5.2`). Default: `gemini-3-pro-preview`.
*   `--mode <str>`:
    *   `graphprior`: Uses GraphPrior 4-phase pipeline.
    *   `llmprior`: Standard LLM prioritization.
    *   `simple`: Simplified prompt.
    *   `direct`: Direct prioritization without tree construction.
*   `--repeat <int>`: Number of times to repeat the experiment. Default: 1.
*   `--file <path>`: Specify a custom CSV file path (overrides `--app`).

### 3. Output
*   **Responses**: Raw LLM outputs are saved in `response/`.
*   **Results**: Parsed prioritization sequences are saved in `result/`.
*   **Console**: The script prints the process status and final APSD / NDCG scores.

## Code Structure

*   **`main.py`**: Entry point. Orchestrates Query -> Parse -> Evaluate workflow.
*   **`kg_service.py`**: Handles Neo4j connection, vector embeddings, and hybrid graph retrieval logic.
*   **`llm.py`**: Interface for Google Gemini API.
*   **`prompt.py`**: Manages prompt templates and context injection.
*   **`parse.py`**: Parses the LLM's hierarchical tree output into a linear sequence.
*   **`config.py`**: Configuration loader.

## License

[MIT License](LICENSE)