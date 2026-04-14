"""
KG-RAG 增强版主流程（修订版）

四阶段流水线：
  Phase 1 - 实体提取（extractor）       → reports[i]["entities"]
  Phase 2 - KG 扩充 + 图匹配分类        → reports[i]["kg_class"]
  Phase 3 - 分类后严重性推导（severity） → reports[i]["severity"]
  Phase 4 - LLM 最终排序（prompt_kg）   → response/{app}-kg-{i}.txt
           + 解析（parse_v3）            → result/{app}-kg-{i}.txt
           + APFD 评估                   → 打印均值

Mode 说明：
  "kg"      — 完整四阶段（推荐）
  "kg-only" — 仅 Phase1~3，用 KG 贪心排序，不调 LLM（最快，作 baseline）

原有 mode（default/simple/direct）不受影响，从 main.py 调用。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from openai import OpenAI

from kg.graph import BugKnowledgeGraph
from kg.extractor import extract_all
from kg.classifier import classify_all, llm_fallback_classify
from kg.severity import compute_all_severities
from prompt_kg import get_kg_prompt
from prompt import load_reports
from llm import ask_gpt
from parse import extract_report_num
from main import REPORTS_MSG

_CLIENT = OpenAI()

# ──────────────────────────────────────────────────────────────────────────────
# 工具：解析编号列表（与 parse_v3 逻辑等价，直接内联避免依赖 mode 参数）
# ──────────────────────────────────────────────────────────────────────────────

def _parse_numbered_list(s: str) -> str:
    """
    解析 LLM 输出的编号列表：
      1. Report 5
      2. Report 12
      ...
    返回格式与原 result 文件一致：
      Report 5:
      Report 12:
      ...
    """
    lines = [l.strip() for l in s.split("\n")]
    start = -1
    for i, line in enumerate(lines):
        if line.startswith("1.") and "Report" in line:
            start = i
            break
    if start == -1:
        # 兜底：尝试任意含 "Report" 的行
        start = 0

    seq = []
    for line in lines[start:]:
        if "Report" not in line:
            continue
        try:
            n = extract_report_num(line)
            if n not in seq:
                seq.append(n)
        except Exception:
            continue
    return "\n".join(f"Report {n}:" for n in seq)


# ──────────────────────────────────────────────────────────────────────────────
# KG-only 贪心排序（不调 LLM，用于 baseline 对比）
# ──────────────────────────────────────────────────────────────────────────────

def _kg_greedy_sort(reports: List[dict], kg: BugKnowledgeGraph) -> str:
    """
    贪心策略：
      1. 优先选出每个 bug_type 中 severity 最高的代表报告（保证多样性）
      2. 其余报告按 severity score 降序追加
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for i, r in enumerate(reports):
        bt = r.get("kg_class", {}).get("bug_type", "功能缺陷")
        groups[bt].append((i, r))

    # 每组取 severity priority 最高的报告作为"代表"
    representatives = []
    rest = []
    for bt, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda x: x[1].get("severity", {}).get("priority", 0),
            reverse=True,
        )
        representatives.append(items_sorted[0])
        rest.extend(items_sorted[1:])

    # 代表按 severity priority 降序；其余同样降序
    representatives.sort(key=lambda x: x[1].get("severity", {}).get("priority", 0), reverse=True)
    rest.sort(key=lambda x: x[1].get("severity", {}).get("priority", 0), reverse=True)

    seq = [i + 1 for i, _ in representatives] + [i + 1 for i, _ in rest]
    return "\n".join(f"Report {n}:" for n in seq)


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    app: int,
    repeat: int = 1,
    mode: str = "kg",
    given_range: Optional[List[int]] = None,
    force_refresh: bool = False,
) -> None:
    """
    完整四阶段流水线。

    Args:
        app:           应用编号
        repeat:        重复次数（仅影响 Phase 4 LLM 调用次数）
        mode:          "kg" | "kg-only"
        given_range:   指定重复编号列表，None 则 1..repeat
        force_refresh: 重新执行 Phase 1 实体提取（忽略缓存）
    """
    print(f"\n=== APP-{app} [{mode}] ===")

    # ── Phase 1：实体提取 ──────────────────────────────────────────────────────
    reports = load_reports(app=app)
    reports = extract_all(reports, app=app, force_refresh=force_refresh)

    # ── Phase 2：KG 扩充 + 图匹配分类 ─────────────────────────────────────────
    kg = BugKnowledgeGraph()
    kg.enrich_from_extractions(reports)           # 将 specific_pages/elements 加入图
    classify_all(reports, kg)                     # 图匹配打分
    llm_fallback_classify(reports, _CLIENT)       # 兜底：对无匹配报告用 LLM 分类

    print(f"[graph]  {kg.summary()}")

    # ── Phase 3：分类后严重性推导 ──────────────────────────────────────────────
    severities = compute_all_severities(reports, kg)
    for bt, (lbl, score) in sorted(severities.items(), key=lambda x: -x[1][1]):
        count = sum(1 for r in reports if r.get("kg_class", {}).get("bug_type") == bt)
        print(f"  {bt:12s} → {lbl}({score:.2f})  [{count} reports]")

    # ── Phase 4：LLM 排序 / KG 贪心排序 ───────────────────────────────────────
    repeats = given_range or list(range(1, repeat + 1))

    if mode == "kg-only":
        result_str = _kg_greedy_sort(reports, kg)
        for i in repeats:
            with open(f"result/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f:
                f.write(result_str)
        print(f"[kg-only] Done.")
        return

    # "kg" mode：调用 LLM
    prompt = get_kg_prompt(reports, kg)
    for i in repeats:
        print(f"  [LLM] iteration-{i}...")
        response, token_p, token_r = ask_gpt(prompt)
        with open(f"response/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f:
            f.write(f"{response}\n{token_p}\n{token_r}")
        # 立即解析，写 result
        try:
            result = _parse_numbered_list(response)
            with open(f"result/{app}-{mode}-{i}.txt", encoding="utf-8", mode="w") as f:
                f.write(result)
        except Exception as e:
            print(f"  [parse] Error: {e}. Please manually fix result/{app}-{mode}-{i}.txt")


def run_evaluate(
    app: int,
    mode: str = "kg",
    repeat: int = 1,
    print_each: bool = False,
) -> float:
    """
    基于 APFD 指标评估排序结果（与原始 run_evaluate 完全相同的计算逻辑）。
    """
    apfds = []
    bug_type_set: set = set()
    total          = REPORTS_MSG[app][0]
    bug_type_count = REPORTS_MSG[app][1]
    reports        = load_reports(app=app)

    for i in range(1, repeat + 1):
        try:
            with open(f"result/{app}-{mode}-{i}.txt", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"[eval] result/{app}-{mode}-{i}.txt not found, skipping.")
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
            # 未被覆盖的缺陷类型视作排在末尾才被发现（最差情况惩罚）
            print(f"[eval] iteration {i}: {missing} bug type(s) not covered, applying tail penalty")
            sigma += missing * total
        apfd = 1 - sigma / (total * bug_type_count) + 1 / (2 * total)
        if print_each:
            print(f"  [{i}] APFD = {apfd:.4f}")
        apfds.append(apfd)
        bug_type_set.clear()

    if not apfds:
        print("[eval] No valid results.")
        return 0.0

    mean_val = float(np.mean(apfds))
    print(f"Mean APFD [{mode}]: {np.round(mean_val, 4)}")
    return mean_val


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    APP    = 7      # 应用编号（对应 REPORTS_MSG 下标）
    REPEAT = 8       # 重复次数（建议 ≥3 以稳定均值）
    MODE   = "kg"    # "kg" | "kg-only"

    run_pipeline(app=APP, repeat=REPEAT, mode=MODE)
    run_evaluate(app=APP, repeat=REPEAT, mode=MODE, print_each=True)
