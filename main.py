"""
Main entry point for Bug Report Prioritization using LLM.

Usage:
    python main.py --app 21 --mode rag --repeat 1
    python main.py --file reports/app21.csv --skip-query --skip-parse
"""

from typing import List, Optional
import os
import argparse
import numpy as np

from config import RESPONSE_DIR, RESULT_DIR
from prompt import get_prompt, load_reports, load_reports_from_file
from llm import ask_gpt
from parse import parse, extract_report_num


def run_query(app: int, repeat: int = 1, mode: str = "default", 
              given_range: Optional[List[int]] = None, csv_file: Optional[str] = None) -> None:
    """Query LLM to categorize bug reports."""
    print(f"Mode: {mode}")
    print(f"APP-{app}...")
    prompt = get_prompt(app=app, mode=mode, csv_file=csv_file)
    
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    
    for i in repeats:
        print(f"iteration-{i}...")
        response, token_usage_p, token_usage_r = ask_gpt(prompt)
        response_file = os.path.join(RESPONSE_DIR, f"{app}-{mode}-{i}.txt")
        with open(response_file, encoding="utf-8", mode="w") as f:
            f.write(f"{response}\nInput Tokens: {token_usage_p}\nOutput Tokens: {token_usage_r}")


def run_parse(app: int, mode: str = "default", repeat: int = 1, 
              given_range: Optional[List[int]] = None, use_severity: bool = False, 
              csv_file: Optional[str] = None) -> None:
    """Parse LLM response to generate prioritized sequence."""
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    
    for i in repeats:
        response_file = os.path.join(RESPONSE_DIR, f"{app}-{mode}-{i}.txt")
        result_file = os.path.join(RESULT_DIR, f"{app}-{mode}-{i}.txt")
        
        try:
            with open(response_file, encoding="utf-8", mode="r") as f1:
                s = f1.read()
        except Exception as e:
            print(f"Error in reading response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        
        try:
            result = parse(s, mode=mode, app=app, use_severity=use_severity, csv_file=csv_file)
        except Exception as e:
            print(f"Error in parsing response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        
        try:
            with open(result_file, encoding="utf-8", mode="w") as f2:
                f2.write(result)
        except Exception as e:
            print(f"Error in writing result {app}-{mode}-{i}: {e}. Please manually check.")
            continue


def run_evaluate(app: int, mode: str = "default", repeat: int = 1, 
                 csv_file: Optional[str] = None) -> None:
    """Evaluate APFD based on bug_ID from CSV file.
    
    APFD = 1 - (sum of first occurrence positions for each bug_ID) / (n * m) + 1/(2n)
    where n = total number of reports, m = number of unique bug types
    """
    apfds = []
    
    # Load reports from CSV file to get bug_ID mapping
    if csv_file:
        reports = load_reports_from_file(csv_file)
    else:
        reports = load_reports(app=app)
    
    total = len(reports)  # n: total number of reports
    
    # Get unique bug_IDs (bug types) from the CSV
    bug_ids = set()
    for report in reports:
        bug_id = report.get("bug_id") or report.get("bug_type")
        if bug_id:
            bug_ids.add(bug_id)
    bug_type_count = len(bug_ids)  # m: number of unique bug types
    
    print(f"Reports: {total}, Unique Bug Types: {bug_type_count}")
    
    for i in range(1, repeat + 1):
        bug_type_set = set()
        result_file = os.path.join(RESULT_DIR, f"{app}-{mode}-{i}.txt")
        
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                sigma = 0
                lines = f.readlines()
                valid_position = 0
                
                for line in lines:
                    try:
                        report_index = extract_report_num(line=line) - 1
                    except:
                        continue
                    
                    if report_index < 0 or report_index >= len(reports):
                        continue
                    
                    valid_position += 1
                    bug_id = reports[report_index].get("bug_id") or reports[report_index].get("bug_type")
                    
                    if bug_id and bug_id not in bug_type_set:
                        bug_type_set.add(bug_id)
                        sigma += valid_position
                
                if len(bug_type_set) != bug_type_count:
                    print(f"Warning: Iteration {i} covered only {len(bug_type_set)}/{bug_type_count} bug types. Skipping APFD calculation for this run.")
                    continue
                
                # APFD formula
                apfd = 1 - (sigma / (total * bug_type_count)) + (1 / (2 * total))
                print(f"Iteration {i}: APFD = {apfd:.4f}")
                apfds.append(apfd)
                
        except FileNotFoundError:
            print(f"Result file for {app}-{mode}-{i} not found.")
            continue
    
    if apfds:
        mean_val = np.mean(apfds)
        print(f"Mean APFD Value: {np.round(mean_val, 4)}")
    else:
        print("No valid results to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bug Report Prioritization using LLM")
    parser.add_argument("--app", type=int, default=21, help="App number (default: 21)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions (default: 1)")
    parser.add_argument("--mode", type=str, default="rag", 
                        choices=["default", "rag", "simple", "direct"], 
                        help="Mode (default: default)")
    parser.add_argument("--file", type=str, default=None, 
                        help="CSV file path (default: reports/app{APP}.csv)")
    parser.add_argument("--severity", action="store_true", 
                        help="Enable severity-aware sorting")
    parser.add_argument("--skip-query", action="store_true", 
                        help="Skip LLM query step")
    parser.add_argument("--skip-parse", action="store_true", 
                        help="Skip parsing step")
    
    args = parser.parse_args()
    
    csv_file = args.file
    
    # Ensure directories exist
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    if not args.skip_query:
        run_query(app=args.app, repeat=args.repeat, mode=args.mode, 
                  given_range=None, csv_file=csv_file)
    
    if not args.skip_parse:
        run_parse(app=args.app, repeat=args.repeat, mode=args.mode, 
                  given_range=None, use_severity=args.severity, csv_file=csv_file)
    
    run_evaluate(app=args.app, repeat=args.repeat, mode=args.mode, csv_file=csv_file)
