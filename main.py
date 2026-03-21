"""
Main entry point for Bug Report Prioritization using LLM.

Usage:
    python main.py --app 21 --mode graphprior --repeat 1
    python main.py --file reports/app21.csv --skip-query --skip-parse
"""

from typing import List, Optional
import os
import argparse
import re
import numpy as np

from config import RESPONSE_DIR, RESULT_DIR, BASE_URL
from prompt import get_prompt
from llm import ask_gpt
from parse import parse
from embedding import run_embedding_mode, compute_apsd, deduplicate_reports
from graph_prior import run_graph_prior
from report_utils import (
    resolve_repeats,
    get_response_path,
    get_result_path,
    resolve_reports,
    extract_sequence_from_lines,
    compute_apsd_from_sequence,
    compute_apfd_from_sequence,
    compute_ndcg_from_sequence,
    ensure_report_scores,
)


def run_query(app: int, repeat: int = 1, mode: str = "llmprior", model: str = "gemini-3-pro-preview",
              given_range: Optional[List[int]] = None, csv_file: Optional[str] = None,
              embedding_model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None,
              kg_mode: str = "offline") -> None:
    """Query LLM to categorize bug reports."""

    
    print(f"Mode: {mode}")
    print(f"APP-{app}...")
    print(f"Model: {model}")
    
    prompt = get_prompt(app=app, mode=mode, csv_file=csv_file, model=model, embedding_model=embedding_model, base_url=base_url, api_key=api_key, kg_mode=kg_mode)
    
    for i in resolve_repeats(repeat=repeat, given_range=given_range):
        print(f"iteration-{i}...")
        response, token_usage_p, token_usage_r = ask_gpt(
            prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        response_file = get_response_path(app=app, mode=mode, iteration=i, model=model)
        with open(response_file, encoding="utf-8", mode="w") as f:
            f.write(f"{response}\nInput Tokens: {token_usage_p}\nOutput Tokens: {token_usage_r}")


def run_parse(app: int, mode: str = "llmprior", repeat: int = 1, model: str = "gemini-3-pro-preview",
              given_range: Optional[List[int]] = None, use_severity: bool = True, 
              csv_file: Optional[str] = None, kg_mode: str = "offline") -> None:
    """Parse LLM response to generate prioritized sequence."""
    for i in resolve_repeats(repeat=repeat, given_range=given_range):
        response_file = get_response_path(app=app, mode=mode, iteration=i, model=model)
        result_file = get_result_path(app=app, mode=mode, iteration=i, model=model)
        
        try:
            with open(response_file, encoding="utf-8", mode="r") as f1:
                s = f1.read()
        except Exception as e:
            print(f"Error in reading response {app}-{mode}-{i}-{model}: {e}. Please manually check.")
            continue
        
        try:
            result = parse(s, mode=mode, app=app, use_severity=use_severity, csv_file=csv_file, kg_mode=kg_mode)
        except Exception as e:
            print(f"Error in parsing response {app}-{mode}-{i}-{model}: {e}. Please manually check.")
            continue
        
        try:
            with open(result_file, encoding="utf-8", mode="w") as f2:
                f2.write(result)
        except Exception as e:
            print(f"Error in writing result {app}-{mode}-{i}-{model}: {e}. Please manually check.")
            continue


def run_embedding(app: int, model: str = "gemini-3-pro-preview", csv_file: Optional[str] = None,
                  thresholds: Optional[List[float]] = None, embedding_model: Optional[str] = None,
                  base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
    """Run embedding-based deduplication mode with multiple thresholds.
    
    Computes embeddings for all reports, evaluates APSD/NDCG at each similarity
    threshold, and reports the optimal threshold.
    """
    # Load reports
    reports = resolve_reports(app=app, csv_file=csv_file)
    
    if thresholds is None:
        thresholds = [round(0.70 + i * 0.05, 2) for i in range(7)]  # 0.70 to 1.00
    
    print(f"\n{'='*60}")
    print(f"Embedding Mode — APP-{app}, Model: {model}")
    print(f"Reports: {len(reports)}, Thresholds: {thresholds}")
    print(f"{'='*60}")
    
    sim_matrix, results = run_embedding_mode(
        reports,
        thresholds,
        model=model,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )
    
    ensure_report_scores(reports)

    # Evaluate APSD/NDCG for each threshold
    print(f"\n{'─'*60}")
    print(f"{'Threshold':>10}  {'Unique':>6}  {'Dup':>5}  {'APSD':>8}  {'NDCG@10':>8}")
    print(f"{'─'*60}")
    
    best_apsd = -1.0
    best_threshold = None
    metric_results = []
    
    for t in thresholds:
        seq = results[t]
        apsd = compute_apsd(seq, reports)
        ndcg10 = compute_ndcg_from_sequence(seq, reports, top_k=10)
        # Count unique vs dup from the sequence logic
        u_idx, d_idx = deduplicate_reports(sim_matrix, threshold=t)
        
        apsd_str = f"{apsd:.4f}" if apsd is not None else "N/A"
        ndcg_str = f"{ndcg10:.4f}" if ndcg10 is not None else "N/A"
        print(f"{t:>10.2f}  {len(u_idx):>6}  {len(d_idx):>5}  {apsd_str:>8}  {ndcg_str:>8}")
        
        if apsd is not None:
            metric_results.append((t, apsd, ndcg10, len(u_idx), len(d_idx)))
            if apsd > best_apsd:
                best_apsd = apsd
                best_threshold = t
    
    print(f"{'─'*60}")
    
    if best_threshold is not None:
        print(f"\n★ Best APSD = {best_apsd:.4f}  (threshold = {best_threshold:.2f})")
        
        # Save the best result
        best_seq = results[best_threshold]
        result_file = os.path.join(RESULT_DIR, f"{app}-embedding-best-{model}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            for rn in best_seq:
                score = reports[rn - 1].get("severity_score", 1)
                f.write(f"Report {rn} [Score: {float(score):.2f}]:\n")
        print(f"Best sequence saved to {result_file}")
        
        # Also save all threshold results
        summary_file = os.path.join(RESULT_DIR, f"{app}-embedding-summary-{model}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Embedding Mode Summary — APP-{app}, Model: {model}\n")
            f.write(f"{'Threshold':>10}  {'Unique':>6}  {'Dup':>5}  {'APSD':>8}  {'NDCG@10':>8}\n")
            for t, apsd_val, ndcg_val, n_u, n_d in metric_results:
                ndcg_txt = f"{ndcg_val:>8.4f}" if ndcg_val is not None else f"{'N/A':>8}"
                f.write(f"{t:>10.2f}  {n_u:>6}  {n_d:>5}  {apsd_val:>8.4f}  {ndcg_txt}\n")
            f.write(f"\nBest: threshold={best_threshold:.2f}, APSD={best_apsd:.4f}\n")
        print(f"Summary saved to {summary_file}")
    else:
        print("\nNo valid APSD results.")


def run_graphprior(app: int, repeat: int = 1, model: str = "gemini-3-pro-preview", csv_file: Optional[str] = None,
                   batch_size: int = 8, workers: int = 4, embedding_model: Optional[str] = None,
                   base_url: Optional[str] = None, api_key: Optional[str] = None, kg_mode: str = "offline") -> None:
    """Run GraphPrior pipeline and persist ranked sequence/result files."""
    reports = resolve_reports(app=app, csv_file=csv_file)
    if not reports:
        print("No reports found. Skip GraphPrior.")
        return

    for i in range(1, repeat + 1):
        print(f"GraphPrior iteration-{i}...")
        sequence, score_map, p_tokens, r_tokens, raw_responses = run_graph_prior(
            reports=reports,
            model=model,
            batch_size=batch_size,
            workers=workers,
            kg_mode=kg_mode,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
        )

        result_file = get_result_path(app=app, mode="graphprior", iteration=i, model=model)
        with open(result_file, "w", encoding="utf-8") as f:
            for report_num in sequence:
                score = score_map.get(report_num, 1.0)
                reports[report_num - 1]["severity_score"] = score
                f.write(f"Report {report_num} [Score: {score:.2f}]:\n")

        response_file = get_response_path(app=app, mode="graphprior", iteration=i, model=model)
        with open(response_file, "w", encoding="utf-8") as f:
            f.write("GraphPrior executed with 3 phases: extraction/alignment & clustering, component-based severity scoring, diversity ranking.\n")
            f.write(f"Sequence length: {len(sequence)}\n")
            f.write(f"Input Tokens: {p_tokens}\nOutput Tokens: {r_tokens}\n")
            if raw_responses:
                f.write("\n=== Raw Model Responses ===\n")
                for item in raw_responses:
                    report_ids = ", ".join(item.get("report_ids", [])) or "N/A"
                    f.write("\n---\n")
                    f.write(f"Batch Index: {item.get('batch_index', -1)}\n")
                    f.write(f"Call Type: {item.get('call_type', 'unknown')}\n")
                    f.write(f"Report IDs: {report_ids}\n")
                    f.write(f"Batch Input Tokens: {item.get('prompt_tokens', 0)}\n")
                    f.write(f"Batch Output Tokens: {item.get('output_tokens', 0)}\n")
                    f.write("Raw Response:\n")
                    f.write(f"{item.get('content', '')}\n")

        print(f"GraphPrior result saved to {result_file}")


def run_evaluate(app: int, mode: str = "llmprior", model: str = "gemini-3-pro-preview", repeat: int = 1, 
                 csv_file: Optional[str] = None) -> None:
    """Evaluate APSD, APFD, NDCG, and token consumption on ranked result files."""
    apsd_values = []
    apfd_values = []
    ndcg10_values = []
    ndcg20_values = []
    input_tokens_values = []
    output_tokens_values = []
    
    reports = resolve_reports(app=app, csv_file=csv_file)
    ensure_report_scores(reports)
    total = len(reports)

    print(f"Reports: {total}")
    
    for i in range(1, repeat + 1):
        result_file = get_result_path(app=app, mode=mode, iteration=i, model=model)
        response_file = get_response_path(app=app, mode=mode, iteration=i, model=model)
        
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                sequence = extract_sequence_from_lines(lines)
                apsd = compute_apsd_from_sequence(sequence=sequence, reports=reports)
                apfd = compute_apfd_from_sequence(sequence=sequence, reports=reports)
                ndcg10 = compute_ndcg_from_sequence(sequence=sequence, reports=reports, top_k=10)
                ndcg20 = compute_ndcg_from_sequence(sequence=sequence, reports=reports, top_k=20)

                if apsd is None:
                    print(f"Warning: Iteration {i} has invalid sequence for APSD. Skipping this run.")
                    continue

                ndcg10_str = f"{ndcg10:.4f}" if ndcg10 is not None else "N/A"
                ndcg20_str = f"{ndcg20:.4f}" if ndcg20 is not None else "N/A"
                apfd_str = f"{apfd:.4f}" if apfd is not None else "N/A"
                print(f"Iteration {i}: APSD = {apsd:.4f}, APFD = {apfd_str}, NDCG@10 = {ndcg10_str}, NDCG@20 = {ndcg20_str}")
                apsd_values.append(apsd)
                if apfd is not None:
                    apfd_values.append(apfd)
                if ndcg10 is not None:
                    ndcg10_values.append(ndcg10)
                if ndcg20 is not None:
                    ndcg20_values.append(ndcg20)
                
        except FileNotFoundError:
            print(f"Result file for {app}-{mode}-{i}-{model} not found.")
            continue
            
        try:
            with open(response_file, "r", encoding="utf-8") as f:
                content = f.read()
                i_tokens_match = re.search(r"Input Tokens:\s*(\d+)", content)
                o_tokens_match = re.search(r"Output Tokens:\s*(\d+)", content)
                if i_tokens_match:
                    input_tokens_values.append(int(i_tokens_match.group(1)))
                if o_tokens_match:
                    output_tokens_values.append(int(o_tokens_match.group(1)))
        except FileNotFoundError:
            pass
    
    if apsd_values:
        print(f"Mean APSD Value: {np.round(np.mean(apsd_values), 4)}")
        if apfd_values:
            print(f"Mean APFD Value: {np.round(np.mean(apfd_values), 4)}")
        if ndcg10_values:
            print(f"Mean NDCG@10 Value: {np.round(np.mean(ndcg10_values), 4)}")
        if ndcg20_values:
            print(f"Mean NDCG@20 Value: {np.round(np.mean(ndcg20_values), 4)}")
        if input_tokens_values:
            print(f"Mean Input Tokens: {np.round(np.mean(input_tokens_values), 1)}")
        if output_tokens_values:
            print(f"Mean Output Tokens: {np.round(np.mean(output_tokens_values), 1)}")
        if input_tokens_values and output_tokens_values:
            total_tokens = np.array(input_tokens_values) + np.array(output_tokens_values)
            print(f"Mean Total Tokens: {np.round(np.mean(total_tokens), 1)}")
    else:
        print("No valid results to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug Report Prioritization using LLM")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", 
                        help="Model name (default: gemini-3-pro-preview). Support Google Gemini and OpenAI models.")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model name. If omitted, provider-specific default is used.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI-compatible base URL for custom model endpoints.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for OpenAI/OpenAI-compatible endpoints.")
    parser.add_argument("--kg-mode", type=str, default="offline", choices=["offline", "online"],
                        help="Knowledge graph backend. Default uses local offline KG built from KG_desc.md.")
    parser.add_argument("--app", type=int, default=1, help="App number (default: 1)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions (default: 1)")
    parser.add_argument("--mode", type=str, default="graphprior", 
                        choices=["llmprior", "graphprior", "simple", "direct", "embedding"], 
                        help="Mode (default: graphprior). 'embedding' uses cosine similarity for deduplication.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="Similarity thresholds for embedding mode (default: 0.70 to 1.00 step 0.05)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for GraphPrior extraction requests")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker threads for GraphPrior chunk processing")
    parser.add_argument("--file", type=str, default=None, 
                        help="CSV file path (default: reports/app{APP}.csv)")
    parser.add_argument("--skip-query", action="store_true", 
                        help="Skip LLM query step")
    parser.add_argument("--skip-parse", action="store_true", 
                        help="Skip parsing step")
    
    args = parser.parse_args()
    
    csv_file = args.file
    base_url = args.base_url or BASE_URL
    
    # Ensure directories exist
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    if args.mode == "embedding":
        # Embedding mode: standalone pipeline, no LLM query/parse needed
        run_embedding(app=args.app, model=args.model, csv_file=csv_file,
                      thresholds=args.thresholds, embedding_model=args.embedding_model,
                      base_url=base_url, api_key=args.api_key)
    elif args.mode == "graphprior":
        # GraphPrior mode: standalone 3-phase pipeline
        run_graphprior(app=args.app, repeat=args.repeat, model=args.model, csv_file=csv_file,
                       batch_size=args.batch_size, workers=args.workers, embedding_model=args.embedding_model,
                       base_url=base_url, api_key=args.api_key, kg_mode=args.kg_mode)
        run_evaluate(app=args.app, repeat=args.repeat, mode="graphprior", model=args.model, csv_file=csv_file)
    else:
        if not args.skip_query:
            run_query(app=args.app, repeat=args.repeat, mode=args.mode, model=args.model,
                      given_range=None, csv_file=csv_file, embedding_model=args.embedding_model,
                      base_url=base_url, api_key=args.api_key, kg_mode=args.kg_mode)
        
        if not args.skip_parse:
            run_parse(app=args.app, repeat=args.repeat, mode=args.mode, 
                      given_range=None, model=args.model, use_severity=True, csv_file=csv_file, kg_mode=args.kg_mode)
        
        run_evaluate(app=args.app, repeat=args.repeat, mode=args.mode, model=args.model, csv_file=csv_file)
