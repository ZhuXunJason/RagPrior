from typing import List, Optional

import numpy as np

from prompt import get_prompt, load_reports
from llm import ask_gpt
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


def run_query(app: int, repeat: int = 1, mode: str = "default", given_range: Optional[List[int]] = None, model: str = "gpt-5.4") -> None:
    print(f"APP-{app}...")
    prompt = get_prompt(app=app, mode=mode)
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    for i in repeats:
        print(f"iteration-{i}...")
        response, token_usage_p, token_usage_r = ask_gpt(prompt, model=model)
        with open(f"response/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f:
            f.write(f"{response}\n{token_usage_p}\n{token_usage_r}")


def run_parse(app: int, mode: str = "default", repeat: int = 1, given_range: Optional[List[int]] = None) -> None:
    if given_range is None:
        repeats = [i for i in range(1, repeat + 1)]
    else:
        repeats = given_range
    for i in repeats:
        try:
            with open(f"response/{app}-{mode}-{i}.txt", encoding="utf-8", mode="r") as f1:
                s = f1.read()
        except Exception as e:
            print(f"Error in reading response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        try:
            result = parse(s, mode=mode)
        except Exception as e:
            print(f"Error in parsing response {app}-{mode}-{i}: {e}. Please manually check.")
            continue
        try:
            with open(f"result/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f2:
                f2.write(result)
        except Exception as e:
            print(f"Error in writing result {app}-{mode}-{i}: {e}. Please manually check.")
            continue


def run_evaluate(app: int, mode: str = "default", repeat: int = 1, print_each: bool = False) -> None:
    apfds = []
    bug_type_set = set()
    total = REPORTS_MSG[app][0]
    bug_type_count = REPORTS_MSG[app][1]
    reports = load_reports(app=app)
    for i in range(1, repeat + 1):
        seq = []
        with open(f"result/{app}-{mode}-{i}.txt", "r", encoding="utf-8") as f:
            fix = 0
            sigma = 0
            lines = f.readlines()
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
            assert len(bug_type_set) == bug_type_count, \
                f"{i}, Bug type count mismatch: {len(bug_type_set)} != {bug_type_count}"
            apfd = 1 - (sigma / (total * bug_type_count)) + (1 / (2 * total))
            if print_each:
                print(i, apfd)
            apfds.append(apfd)
            bug_type_set.clear()
    mean_val = np.mean(apfds)
    print(f"Mean APFD Value: {np.round(mean_val, 4)}")


if __name__ == "__main__":
    APP = 1              # set app number
    REPEAT = 3            # set repeat times
    MODE = "default"      # set mode: "default" -> LLMPrior, "simple" -> SimpleLLMPrior, "direct" -> DirectLLMPrior
    MODEL = "gpt-5.4"       # LLM model name

    run_query(app=APP, repeat=REPEAT, mode=MODE, given_range=None, model=MODEL)
    run_parse(app=APP, repeat=REPEAT, mode=MODE, given_range=None)
    run_evaluate(app=APP, repeat=REPEAT, mode=MODE)
