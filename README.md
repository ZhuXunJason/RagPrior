# LLM-Based Bug Report Prioritization with GraphPrior

This project utilizes Large Language Models (LLM) with GraphPrior to prioritize crowdsourced test reports. The newly integrated GraphPrior features a highly efficient three-stage pipeline: extraction, alignment & clustering, followed by affected-components-based severity scoring, and finally diversity-first ranking.

[中文文档](README_CN.md)

## Features

*   **GraphPrior 3-Phase Integrated Pipeline**: Streamed extraction/alignment, exact component-based severity scoring, and diversity ranking.
*   **Multi-Dimensional Dynamic Clustering**: Extracts Operation, Page, Element, and Bug dimensions to detect duplicate candidates via vector cosine similarity, forming precise bug signatures.
*   **Objectively Unbiased Scoring**: Calculates severity by exact affected component counts and occurrence frequency, dropping unstable LLM absolute graph densities, to guarantee scoring stability across models.
*   **Automated Evaluation**: Calculates APSD and NDCG to measure prioritization effectiveness.

## Detailed GraphPrior 3-Phase Workflow

The refactored GraphPrior streamlines extraction, clustering, scoring, and sorting into three highly cohesive stages.

### Phase 1: Extraction, Alignment & Clustering

**Core Files**: `graph_prior.py` (`phase1_extract_and_fuse` function), relying on `llm.py` (LLM invocations) and `embedding.py` (vector embeddings).

*   **Input**: Raw bug report text (title, description, reproduction steps, etc.).
*   **Process**:
    1. Through the LLM, the system structurally extracts four semantic features/dimensions: `Operation`, `Page`, `Element`, and `Bug`. It also assesses the global collision level (`Element`, `Page`, or `Program`).
    2. These details are converted into semantic vectors and **independently** soft-matched (Cosine Similarity) over locally mapped knowledge graph nodes. If the similarity passes the threshold, it hooks up to the existing EID; otherwise, it dynamically generates a new EID in the graph.
    3. The 4 retrieved EIDs combine instantly to compose the bug report's unique "4-Dimension Signature".
    4. Cluster Maintenance: It matches this signature. If exact identical, the report merges seamlessly as a duplicate of the matched branch; else, a brand new logical report cluster is born.
*   **Output**: The definitive list of bug report representative clusters aggregated uniquely via multi-dimensions, along with their constructed sub-network structure.
*   **Why it matters**: Zero-leak deduplication happens immediately at Phase 1, entirely avoiding heavy secondary mapping comparisons or duplicate extraction effort downstream.

### Phase 2: Severity Scoring based on Affected Components

**Core Files**: `graph_prior.py` (`phase2_topology_scoring` function).

*   **Input**: The clustered instance list from Phase 1 combined with component connectivity on the graph (`LOCATED_ON` dependency edges).
*   **Process**: Discarding legacy graph node/edge counting blind spots, the absolute Severity Score is now reliably tallied utilizing the exact affected component layer ($I$) and the frequency of occurrence base ($T$).
    The system counts length $T$ for the duplicate cluster, then deduces damaged components count $I$:
    1. If typed as `element` scope locally, only a single node is considered harmed; hence $I = 1$.
    2. If typed as `page` scope branch, the system grabs all valid unique instantiated element components registered under this `page_eid`, returning that specific list count as $I$.
    3. If typed as a global crash/halt (`program` scope limit), $I$ equates to the absolute total volume of pooled interactive components submitted up to this tick.
    **Formula applied**: $Score = I \times \log(T + 1)$
*   **Output**: Accurate integer/float severity proxy score for every independent clustered set.
*   **Why it matters**: Grounded strongly in component radius tracking combined precisely with historical volume overlap computations, generating a purely deterministic, unshakeable grading baseline against fickle LLM whims.

### Phase 3: Diversity-first Ranking

**Core Files**: `graph_prior.py` (`run` method for orchestrating selection), coordinated by `main.py`.

*   **Input**: The summarized duplicate clusters alongside their objectively assigned scores.
*   **Process**: Evaluates the computed descending Danger-Score sorting all clustered representations. The sorted main representatives lead the output sequence. Attached directly/nested behind each representative are their corresponding identical duplicates for pure reference.
*   **Output**: The straightforward 1D sequence mapping to end-users combining exploring novelty priority with duplicate logs.
*   **Why it matters**: Assures that top review queues display high-risk, non-duplicate clusters prominently maximizing budget reach, securely tucking repetitive complaints further down the pipe efficiently.

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