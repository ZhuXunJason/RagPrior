"""
Grid experiment runner: compare APFD across 3 configurations.
  1) main_kg.py  dedup_threshold=1.0  mode=kg
  2) main_kg.py  dedup_threshold=0.9  mode=kg
  3) main.py     mode=default
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_DUPLICATE_THRESHOLD
from main_kg import run_pipeline as kg_run_pipeline
from main_kg import run_evaluate as kg_run_evaluate
from main import run_query, run_parse, REPORTS_MSG
from prompt import load_reports
from parse import extract_report_num


def evaluate_default(app: int, repeat: int = 3) -> float:
    """Replicate main.py run_evaluate but return the mean APFD."""
    apfds = []
    bug_type_set: set = set()
    total = REPORTS_MSG[app][0]
    bug_type_count = REPORTS_MSG[app][1]
    reports = load_reports(app=app)

    for i in range(1, repeat + 1):
        try:
            with open(f"result/{app}-default-{i}.txt", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"  [eval] result/{app}-default-{i}.txt not found, skipping.")
            continue

        fix, sigma = 0, 0
        for index, line in enumerate(lines):
            report_index = extract_report_num(line=line) - 1
            try:
                bug_type = reports[report_index]["bug_type"]
            except IndexError:
                fix -= 1
                continue
            if bug_type not in bug_type_set:
                bug_type_set.add(bug_type)
                sigma += index + fix

        missing = bug_type_count - len(bug_type_set)
        if missing > 0:
            print(f"  [eval] iter {i}: {missing} bug type(s) not covered, tail penalty")
            sigma += missing * total

        apfd = 1 - sigma / (total * bug_type_count) + 1 / (2 * total)
        print(f"  [{i}] APFD = {apfd:.4f}")
        apfds.append(apfd)
        bug_type_set.clear()

    if not apfds:
        return 0.0
    mean_val = float(np.mean(apfds))
    print(f"Mean APFD [default]: {np.round(mean_val, 4)}")
    return mean_val


def main():
    APPS = [1, 2, 3, 4, 5, 6, 7, 8]
    REPEAT = 3

    configs = [
        ("kg_t1.0", "kg", {"dedup_threshold": 1.0, "model": "gpt-5.4"}),
        ("kg_t0.9", "kg", {"dedup_threshold": 0.9, "model": "gpt-5.4"}),
        ("default", "default", {"model": "gpt-5.4"}),
    ]

    # {app: {config_label: mean_apfd}}
    all_results: Dict[int, Dict[str, float]] = {}

    for app in APPS:
        print(f"\n{'='*70}")
        print(f"  APP {app} ({REPORTS_MSG[app][0]} reports, {REPORTS_MSG[app][1]} bug types)")
        print(f"{'='*70}")

        app_results: Dict[str, float] = {}

        for label, mode, kwargs in configs:
            print(f"\n--- Config: {label} ---")

            if mode == "kg":
                # Phase 1: run pipeline (writes to result/{app}-kg-{i}.txt)
                kg_run_pipeline(
                    app=app,
                    repeat=REPEAT,
                    mode="kg",
                    **kwargs,
                )
                # Phase 2: copy result files to unique names
                for i in range(1, REPEAT + 1):
                    src = f"result/{app}-kg-{i}.txt"
                    dst = f"result/{app}-{label}-{i}.txt"
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

                # Phase 3: evaluate
                apfd = kg_run_evaluate(app=app, repeat=REPEAT, mode=label, print_each=True)
                app_results[label] = apfd

            elif mode == "default":
                # Run main.py pipeline
                run_query(app=app, repeat=REPEAT, mode="default", model="gpt-5.4")
                run_parse(app=app, repeat=REPEAT, mode="default")
                # Result files are already at result/{app}-default-{i}.txt
                apfd = evaluate_default(app=app, repeat=REPEAT)
                app_results[label] = apfd

            else:
                raise ValueError(f"Unknown mode: {mode}")

        all_results[app] = app_results

    # ── Print final comparison table ──
    print("\n\n")
    print("=" * 80)
    print("  APFD Comparison Table (avg over 3 repeats)")
    print("=" * 80)

    # Header
    header_cols = ["App", "Reports", "BugTypes", "KG t=1.0", "KG t=0.9", "Default"]
    col_widths = [6, 9, 10, 12, 12, 12]
    header = "  ".join(h.ljust(w) for h, w in zip(header_cols, col_widths))
    print(header)
    print("-" * len(header))

    # Rows
    best_overall = ""
    for app in APPS:
        n_reports = REPORTS_MSG[app][0]
        n_types = REPORTS_MSG[app][1]
        vals = all_results.get(app, {})
        kg10 = vals.get("kg_t1.0", float("nan"))
        kg09 = vals.get("kg_t0.9", float("nan"))
        dflt = vals.get("default", float("nan"))

        row = (
            f"{str(app).ljust(6)}  "
            f"{str(n_reports).ljust(9)}  "
            f"{str(n_types).ljust(10)}  "
            f"{f'{kg10:.4f}'.ljust(12)}  "
            f"{f'{kg09:.4f}'.ljust(12)}  "
            f"{f'{dflt:.4f}'.ljust(12)}"
        )
        print(row)

        # Track best per app
        best_label = max(
            [("KG t=1.0", kg10), ("KG t=0.9", kg09), ("Default", dflt)],
            key=lambda x: x[1] if not np.isnan(x[1]) else float("-inf"),
        )
        best_overall += f"App {app}: best={best_label[0]} ({best_label[1]:.4f})\n"

    print("-" * len(header))

    # Averages across apps
    avg_kg10 = float(np.mean([all_results[a].get("kg_t1.0", float("nan")) for a in APPS]))
    avg_kg09 = float(np.mean([all_results[a].get("kg_t0.9", float("nan")) for a in APPS]))
    avg_dflt = float(np.mean([all_results[a].get("default", float("nan")) for a in APPS]))
    row = (
        f"{'Avg'.ljust(6)}  "
        f"{'—'.ljust(9)}  "
        f"{'—'.ljust(10)}  "
        f"{f'{avg_kg10:.4f}'.ljust(12)}  "
        f"{f'{avg_kg09:.4f}'.ljust(12)}  "
        f"{f'{avg_dflt:.4f}'.ljust(12)}"
    )
    print(row)
    print("=" * 80)

    print("\nBest per app:")
    print(best_overall)


if __name__ == "__main__":
    main()
