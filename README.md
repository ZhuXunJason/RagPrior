# LLMPrior

Prioritize crowdsourced test reports using LLM and KG-RAG.

## Preparation

1. Install requirements: `pip install -r requirements.txt`

2. Create a `.env` file in project root:

```
OPENAI_API_KEY=your_openai_key
BASE_URL=https://api.openai.com/v1
GOOGLE_API_KEY=your_google_key
EMBEDDING_MODEL=gemini-embedding-2-preview
DUPLICATE_THRESHOLD=0.85
```

- `BASE_URL` is optional and can point to a custom OpenAI-compatible endpoint.
- `GOOGLE_API_KEY` is required for KG mode embedding pre-clustering (using Gemini).
- `EMBEDDING_MODEL` and `DUPLICATE_THRESHOLD` are optional KG mode settings.

## Usage

Place `app{n}.csv` files under `./reports/` (set `DATASET_ROOT` in `prompt.py`).

### main.py — LLM-only baseline

```bash
python main.py --app 1 --repeat 3 --mode default --model gpt-5.4
```

Modes: `default` (LLMPrior), `simple` (SimpleLLMPrior), `direct` (DirectLLMPrior).

### main_kg.py — KG-RAG pipeline

```bash
# Full KG pipeline (Phase 1-4)
python main_kg.py --app 7 --repeat 3 --mode kg \
    --model deepseek-v4-flash \
    --dedup-threshold 0.85 \
    --skip-severity

# KG-only greedy sort baseline (no LLM ranking call)
python main_kg.py --app 7 --repeat 3 --mode kg-only --no-skip-severity
```

### Grid experiments

```powershell
# run_exp.ps1: loop through apps 1-8 with two dedup thresholds
.\run_exp.ps1

# Then evaluate:
python eval_table.py
```

## KG-RAG Pipeline (main_kg.py)

The KG-RAG enhanced pipeline consists of four phases:

### Phase 0 — Embedding Pre-clustering (Deduplication)
- Embeds all report descriptions using the configured embedding model (default: `gemini-embedding-2-preview`).
- Groups similar reports by cosine similarity above `DUPLICATE_THRESHOLD`.
- Selects one representative per cluster, reducing the number of reports passed to subsequent phases.
- After LLM ranking, expands the compressed sequence back to all original reports via `expand_ranked_sequence`.

### Phase 1 — Entity Extraction
- For each report, extracts structured entities via LLM: operations, symptoms, specific pages/elements, page type, and module.
- Results are cached in `./cache/app{n}_entities.json` to avoid redundant calls.

### Phase 2 — Knowledge Graph Construction & Classification
- Builds a `BugKnowledgeGraph` from extracted entities — nodes represent pages, operations, symptoms, and modules; edges represent co-occurrence.
- Classifies each report into a bug type using graph pattern matching (TF-IDF weighted).
- Reports with low match confidence fall back to LLM classification.

### Phase 3 — Severity Derivation
- Computes severity scores per bug type based on KG centrality and frequency.
- If `--skip-severity` is set, all reports use neutral severity (priority=0.0), which is used for the `kg` mode's LLM ranking prompt.
- Severity is critical for `kg-only` mode (greedy sort by severity descending).

### Phase 4 — Report Ranking

**`kg` mode (default):** Concatenates representative reports + KG summary into a structured prompt, calls the LLM to produce a prioritized sequence, then expands back to all reports.

**`kg-only` mode:** Greedy sort — picks the highest-severity report from each bug type first (for diversity), then appends remaining reports by severity descending. Zero additional LLM cost.

## Code Structure

| File | Role |
|------|------|
| `main.py` | LLM-only pipeline entry (query, parse, evaluate) |
| `main_kg.py` | KG-RAG pipeline entry (run_pipeline, run_evaluate) |
| `llm.py` | OpenAI-compatible LLM interface |
| `prompt.py` | Prompt templates and report loading |
| `prompt_kg.py` | KG-specific prompt template |
| `parse.py` | Response parsing and sequence extraction |
| `config.py` | Environment config and client factory |
| `kg/extractor.py` | Phase 1: entity extraction |
| `kg/graph.py` | BugKnowledgeGraph data structure |
| `kg/classifier.py` | Phase 2: graph-based bug classification |
| `kg/severity.py` | Phase 3: severity score computation |
| `kg/dedup.py` | Phase 0: embedding-based pre-clustering |
| `eval_table.py` | Evaluate results and print comparison table |
| `run_exp.ps1` | PowerShell script for grid experiments |

## Tips

- If `run_parse` fails, manually fix the LLM response format or rerun specific cases with `given_range`.
- If `run_evaluate` fails, the generated prioritization sequence is invalid — rerun the cases.
- Use `--force-refresh` to re-extract entities (ignoring cache).
