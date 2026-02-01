# LLM-Based Bug Report Prioritization with Graph RAG

This project utilizes Large Language Models (LLM) combined with Knowledge Graph Retrieval-Augmented Generation (Graph RAG) to prioritize crowdsourced test reports. It categorizes reports into a hierarchical tree structure and generates a prioritized sequence to ensure reviewers encounter diverse bug types early.

[中文文档](README_CN.md)

## Features

*   **Graph RAG**: Enhances LLM understanding by retrieving structural context (Operations, Elements) from a Neo4j Knowledge Graph.
*   **Hierarchical Clustering**: Categorizes bugs into fine-grained types using a tree structure.
*   **Diversity-Based Prioritization**: Sorts reports to maximize the variety of bugs reviewed in top positions.
*   **Automated Evaluation**: Calculates APFD (Average Percentage of Faults Detected) to measure prioritization effectiveness.

## Prerequisites

1.  **Neo4j Database**: A running Neo4j instance with the bug knowledge graph populated.
2.  **Google Gemini API Key**: Access to Gemini models (e.g., `gemini-3-flash` or `gemini-embedding-001`).

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Environment Configeration
Open `.env` and fill in your actual credentials. Rename it to `.env`.

### 2. Data Preparation
Place your bug report CSV files in the `reports/` directory (e.g., `reports/app21.csv`).

### 3. Run Prioritization
Use `main.py` to run the prioritization process.

**Basic Command (Graph RAG Mode):**
```bash
python main.py --app 21 --mode rag --repeat 1
```

**Parameters:**
*   `--app <int>`: The App ID (corresponding to `reports/app{ID}.csv`). Default: 21.
*   `--mode <str>`:
    *   `rag`: Uses Knowledge Graph RAG (Recommended).
    *   `default`: Standard LLM prioritization without RAG.
    *   `simple`: Simplified prompt.
    *   `direct`: Direct prioritization without tree construction.
*   --repeat <int>`: Number of times to repeat the experiment. Default: 1.
*   `--file <path>`: Specify a custom CSV file path (overrides `--app`).

### 3. Output
*   **Responses**: Raw LLM outputs are saved in `response/`.
*   **Results**: Parsed prioritization sequences are saved in `result/`.
*   **Console**: The script prints the process status and the final APFD score.

## Code Structure

*   **`main.py`**: Entry point. Orchestrates Query -> Parse -> Evaluate workflow.
*   **`kg_service.py`**: Handles Neo4j connection, vector embeddings, and hybrid graph retrieval logic.
*   **`llm.py`**: Interface for Google Gemini API.
*   **`prompt.py`**: Manages prompt templates and context injection.
*   **`parse.py`**: Parses the LLM's hierarchical tree output into a linear sequence.
*   **`config.py`**: Configuration loader.

## License

[MIT License](LICENSE)