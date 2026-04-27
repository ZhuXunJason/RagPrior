from pathlib import Path
from typing import List, Optional

import numpy as np

from prompt import get_prompt, load_reports
from llm import ask_completion
from parse import parse, extract_report_num

REPORTS_MSG = [
        [],
        [276, 20],  # 1
        [134, 9],  # 2
        [29, 6],  # 3
        [9, 8],  # 4
        [13, 3],  # 5
        [26, 4],  # 6
        [152, 8],  # 7
        [41, 4],  # 8
        [75, 7],  # 9
        [26, 5],  # 10
        [5, 3],  # 11
        [10, 2],  # 12
        [17, 9],  # 13
        [131, 8],  # 14
        [88, 15],  # 15
        [51, 8],  # 16
        [4, 2],  # 17
        [294, 12],  # 18
        [12, 3],  # 19
        [24, 5],  # 20
    ]

LLMPRIOR_MAX_TOKEN = 32768


def run_query(app: int, repeat: int = 1, mode: str = "default", given_range: Optional[List[int]] = None, model: str = "gpt-5.4") -> None:
    print(f"APP-{app}...")
    prompt = get_prompt(app=app, mode=mode)
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    for i in repeats:
        print(f"iteration-{i}...")
        llm_result = ask_completion(
            prompt,
            model=model,
            max_token=LLMPRIOR_MAX_TOKEN,
        )
        response = llm_result.get("content", "")
        token_usage_p = llm_result.get("prompt_tokens", 0)
        token_usage_r = llm_result.get("completion_tokens", 0)
        with open(f"response/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f:
            if response:
                f.write(f"{response}\n")
            f.write(f"Input Tokens: {token_usage_p}\n")
            f.write(f"Output Tokens: {token_usage_r}\n")
            f.write(f"Provider: {llm_result.get('provider', '')}\n")
            f.write(f"Finish Reason: {llm_result.get('finish_reason', '')}\n")


def run_parse(app: int, mode: str = "default", repeat: int = 1, given_range: Optional[List[int]] = None) -> None:
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    for i in repeats:
        result_path = Path(f"result/{app}-{mode}-{i}.txt")
        try:
            with open(f"response/{app}-{mode}-{i}.txt", encoding="utf-8", mode="r") as f1:
                s = f1.read()
        except Exception as e:
            print(f"Error in reading response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        try:
            result = parse(s, mode=mode)
        except Exception as e:
            if result_path.exists():
                result_path.unlink()
            print(f"Error in parsing response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        try:
            with open(result_path, encoding="utf-8", mode="w") as f2:
                f2.write(result)
        except Exception as e:
            print(f"Error in writing result {app}-{mode}-{i}: {e}. Please manually check.")
            continue


def run_evaluate(app: int, mode: str = "default", repeat: int = 1, print_each: bool = True) -> float:
    apfds = []
    bug_type_set = set()
    total = REPORTS_MSG[app][0]
    bug_type_count = REPORTS_MSG[app][1]
    reports = load_reports(app=app)
    for i in range(1, repeat + 1):
        seq = []
        try:
            with open(f"result/{app}-{mode}-{i}.txt", "r", encoding="utf-8") as f:
                fix = 0
                sigma = 0
                lines = f.readlines()
        except FileNotFoundError:
            print(f"[eval] result/{app}-{mode}-{i}.txt not found, skipping.")
            continue
        for index, line in enumerate(lines):
            report_index = extract_report_num(line=line) - 1
            try:
                bug_type = reports[report_index]["bug_type"]
                seq.append(report_index)
            except IndexError:
                fix -= 1
                continue
            if bug_type not in bug_type_set:
                bug_type_set.add(bug_type)
                sigma += index + fix
        missing = bug_type_count - len(bug_type_set)
        if missing > 0:
            print(f"[eval] iteration {i}: {missing} bug type(s) not covered, applying tail penalty")
            sigma += missing * total
        apfd = 1 - (sigma / (total * bug_type_count)) + (1 / (2 * total))
        if print_each:
            print(i, apfd)
        apfds.append(apfd)
        bug_type_set.clear()
    if not apfds:
        print("Mean APFD Value: 0.0")
        return 0.0
    mean_val = float(np.mean(apfds))
    print(f"Mean APFD Value: {np.round(mean_val, 4)}")
    return mean_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMPrior pipeline")
    parser.add_argument("--app", type=int, default=1, help="app number")
    parser.add_argument("--repeat", type=int, default=3, help="repeat times")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "simple", "direct"], help="pipeline mode")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="LLM model name")
    args = parser.parse_args()

    # File naming: {app}-llm-{model}-{i}.txt  e.g. "1-llm-gpt-5.4-1.txt"
    file_tag = f"llm-{args.model}"

    run_query(app=args.app, repeat=args.repeat, mode=file_tag, given_range=None, model=args.model)
    run_parse(app=args.app, repeat=args.repeat, mode=file_tag, given_range=None)
    run_evaluate(app=args.app, repeat=args.repeat, mode=file_tag)
