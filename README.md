# LLM-Based Bug Report Prioritization with GraphPrior

This project utilizes Large Language Models (LLM) with GraphPrior to prioritize crowdsourced test reports. GraphPrior includes four stages: extraction & instance fusion, topology-based severity scoring, hybrid graph/vector de-duplication, and diversity-first ranking.

[中文文档](README_CN.md)

## Features

*   **GraphPrior 4-Phase Pipeline**: extraction/fusion, topology scoring, hybrid de-duplication, and diversity ranking.
*   **Hierarchical Clustering**: Categorizes bugs into fine-grained types using a tree structure.
*   **Diversity-Based Prioritization**: Sorts reports to maximize the variety of bugs reviewed in top positions.
*   **Automated Evaluation**: Calculates APSD and NDCG to measure prioritization effectiveness.

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