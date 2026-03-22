# LLM-Based Bug Report Prioritization with GraphPrior

[中文文档](README_CN.md)

This project prioritizes crowdsourced bug reports with two main modes:

- `llmprior`: direct LLM-based prioritization
- `graphprior`: structure-aware prioritization with offline KG priors, selective LLM linking, clustering, deterministic scoring, and diversity reranking

## What GraphPrior Is Now

The current GraphPrior implementation is no longer the old “free-form extraction + Neo4j-first” pipeline.

It now uses:

- offline KG by default, built from [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)
- relation-enhanced candidate generation
- app-local lexicon mining from the current app’s reports
- deterministic slot prefill before any LLM call
- selective LLM linking only for ambiguous slots
- single-line extraction output such as `R_001|O1|P0|E2|B3|page`
- Qwen no-think request options for GraphPrior extraction
- deterministic scope post-calibration

Neo4j is still supported as an optional backend via `--kg-mode online`, but it is no longer the default path.

## GraphPrior Pipeline

### Phase 1: Candidate Generation and Selective Linking

Core file: [`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

For each representative report, GraphPrior builds candidates for `Operation`, `Page`, `Element`, and `Bug` from three evidence sources:

- current-report local phrase extraction
- app-local lexicon mined from the current app’s reports
- offline KG relation priors from [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)

The candidates are ranked by layered evidence rather than a black-box mixed score:

1. direct report hit
2. app-local lexicon support
3. offline KG support
4. memory-based similarity support

Then GraphPrior performs deterministic prefill:

- no candidate: prefill `NONE` / `未知*`
- one strong candidate or clear winner: lock it locally
- only ambiguous slots go to the LLM

If all slots are resolved locally, that report skips the LLM entirely.

For LLM-linked reports, the model only selects candidate ids and returns one line per report:

```text
R_001|O1|P0|E2|B3|page
```

This keeps output tokens low and makes parsing robust.

### Phase 2: Alignment, Signature Construction, and Clustering

Core file: [`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

After linking, each report is converted into a 4-dimensional signature:

- `operation_eid`
- `page_eid`
- `element_eid`
- `bug_eid`

Each dimension is matched against the in-memory instance store with cosine similarity. If no close instance exists, a new instance node is created.

Two reports are clustered together when their 4-dimensional signatures are identical.

### Phase 3: Deterministic Scoring and Diversity Reranking

Core file: [`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py)

GraphPrior keeps the current deterministic scoring logic:

- `element`: affected component count `I = 1`
- `page`: `I = number of elements attached to that page`
- `program`: `I = total number of known elements`

Score formula:

```text
Score = I * log(T + 1)
```

where `T` is the cluster size.

After scoring, cluster representatives are reranked with a diversity penalty so the head of the ranked list contains fewer near-duplicate issues.

## Why GraphPrior Is Still Useful

Compared with pure `llmprior`, GraphPrior emphasizes:

- lower and more stable output token consumption
- explicit intermediate structure
- auditable candidate evidence
- more reproducible extraction behavior

In practice, this means GraphPrior is designed to trade a small amount of raw LLM flexibility for better interpretability and lower extraction variance.

## Metrics

The evaluation pipeline reports:

- `APFD`: early discovery of different bug groups
- `APSD`: severity-weighted early discovery
- `NDCG@10`, `NDCG@20`: ranking quality at the head

Note:

- when the dataset does not contain external severity ground truth, the current APSD behaves closer to a frequency-weighted proxy than a strict real-world severity metric
- GraphPrior is optimized mainly for `APFD/APSD + token cost`; `NDCG` is useful but secondary

## Requirements

### Required

- Python environment with dependencies from `requirements.txt`
- model API credentials for your chosen LLM backend
- a local [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md) file

### Optional

- Neo4j, only if you want `--kg-mode online`

## Installation

```bash
git clone https://github.com/ZhuXunJason/RAGPrior
cd RAGPrior
pip install -r requirements.txt
```

## Usage

### GraphPrior

Default command:

```bash
python main.py --app 2 --model qwen3.5-plus --embedding-model text-embedding-v4
```

This uses:

- `--mode graphprior` implicitly, because it is the default mode
- `--kg-mode offline` implicitly, because offline KG is the default backend

Optional online KG:

```bash
python main.py --app 2 --model qwen3.5-plus --embedding-model text-embedding-v4 --kg-mode online
```

### LLMPrior

```bash
python main.py --app 2 --mode llmprior --model qwen3.5-plus
```

### Useful CLI Options

- `--app <int>`: app id, corresponding to `reports/app{ID}.csv`
- `--file <path>`: custom CSV path
- `--model <str>`: LLM model name
- `--embedding-model <str>`: embedding model
- `--kg-mode offline|online`: KG backend, default `offline`
- `--batch-size <int>`: GraphPrior batch size
- `--workers <int>`: GraphPrior worker threads
- `--repeat <int>`: number of repeated runs

## Outputs

- [`response/`](/d:/OneDrive/Project/GraphPrior/GraphPrior/response): raw LLM responses and GraphPrior debug info
- [`result/`](/d:/OneDrive/Project/GraphPrior/GraphPrior/result): ranked sequences

GraphPrior response files now include:

- per-batch input/output token usage
- raw model output
- prefill information
- slot evidence summaries
- candidate statistics for debugging extraction quality

## Code Structure

- [`main.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/main.py): entry point and evaluation loop
- [`graph_prior.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/graph_prior.py): GraphPrior pipeline
- [`offline_kg.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/offline_kg.py): offline KG parser and candidate retrieval
- [`kg_service.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/kg_service.py): online/offline KG dispatch
- [`llm.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/llm.py): LLM invocation layer
- [`report_utils.py`](/d:/OneDrive/Project/GraphPrior/GraphPrior/report_utils.py): report loading and metrics
- [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md): offline KG entity and relation definition

## Current Limitations

- GraphPrior is still conservative on `Page / Element` extraction in some apps
- the offline KG is only as good as the coverage of [`KG_desc.md`](/d:/OneDrive/Project/GraphPrior/GraphPrior/KG_desc.md)
- app-local lexicon is heuristic and can still miss rare expressions
- current scoring is deterministic and explainable, but not yet learned from larger cross-app supervision
- results can still vary by model family, especially outside the extraction path

## License

[MIT License](LICENSE)
