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

from config import (
    get_base_url,
    get_duplicate_threshold,
    get_embedding_model,
    get_google_api_key,
    get_openai_api_key,
)
from kg.dedup import (
    deduplicate_reports_by_embedding,
    expand_ranked_sequence,
    select_representative_reports,
)
from kg.graph import BugKnowledgeGraph
from kg.extractor import extract_all
from kg.enricher import enrich_reports
from kg.classifier import classify_all, llm_fallback_classify
from kg.severity import compute_all_severities
from prompt_kg import get_kg_prompt
from prompt import load_reports
from llm import ask_completion
from parse import extract_report_num
from main import REPORTS_MSG

KG_RANKING_MAX_TOKEN = 32768

# ──────────────────────────────────────────────────────────────────────────────
# 工具：解析编号列表（与 parse_v3 逻辑等价，直接内联避免依赖 mode 参数）
# ──────────────────────────────────────────────────────────────────────────────

def _parse_numbered_sequence(s: str) -> List[int]:
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

    seq: List[int] = []
    for line in lines[start:]:
        if "Report" not in line:
            continue
        try:
            n = extract_report_num(line)
            if n not in seq:
                seq.append(n)
        except Exception:
            continue
    return seq


def _format_result_sequence(seq: List[int]) -> str:
    return "\n".join(f"Report {n}:" for n in seq)


def _validate_subset_sequence(seq: List[int], expected_count: int) -> None:
    if not seq:
        raise ValueError("Empty ranking sequence parsed from model response.")
    if expected_count >= 20 and len(seq) < max(5, expected_count // 4):
        raise ValueError(
            f"Ranking sequence too short for {expected_count} reports: {len(seq)}"
        )


def _default_severity() -> dict:
    return {
        "label": "中",
        "priority": 0.0,
        "elem_crit": 0,
    }


def _assign_default_severity(reports: List[dict]) -> None:
    """Fill neutral severity so downstream prompt/sort logic remains stable."""
    for r in reports:
        sev = r.get("severity")
        if not isinstance(sev, dict):
            r["severity"] = _default_severity()
            continue
        if "label" not in sev:
            sev["label"] = "中"
        if "priority" not in sev:
            sev["priority"] = 0.0
        if "elem_crit" not in sev:
            sev["elem_crit"] = 0


def _map_subset_to_original(
    subset_sequence: List[int],
    representative_nums: List[int],
) -> List[int]:
    """Map report numbers in compressed prompt space back to original report numbers."""
    mapped: List[int] = []
    for n in subset_sequence:
        idx = n - 1
        if 0 <= idx < len(representative_nums):
            original_num = representative_nums[idx]
            if original_num not in mapped:
                mapped.append(original_num)

    # Keep robust behavior if LLM drops some compressed reports in output.
    for n in representative_nums:
        if n not in mapped:
            mapped.append(n)
    return mapped


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


def _severity_desc_sequence(reports: List[dict]) -> List[int]:
    """
    Sort by severity priority descending.
    Tie-breakers: KG score desc, then original order.
    """
    order = sorted(
        enumerate(reports),
        key=lambda x: (
            x[1].get("severity", {}).get("priority", 0.0),
            x[1].get("kg_class", {}).get("score", 0.0),
            -(x[0] + 1),
        ),
        reverse=True,
    )
    return [i + 1 for i, _ in order]


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    app: int,
    repeat: int = 1,
    mode: str = "kg",
    output_tag: Optional[str] = None,
    given_range: Optional[List[int]] = None,
    force_refresh: bool = False,
    dedup_threshold: Optional[float] = None,
    embedding_model: Optional[str] = None,
    model: str = "gpt-5.4",
    skip_severity_calculation: bool = False,
    use_severity_sort: bool = False,
    use_vision: bool = False,
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
    if skip_severity_calculation and use_severity_sort:
        raise ValueError(
            "skip_severity_calculation 和 use_severity_sort 不能同时为 True。"
        )

    file_mode = output_tag or mode

    print(f"\n=== APP-{app} [{mode}] ===")

    reports_all = load_reports(app=app)
    reports = reports_all
    dedup_result = None

    # ── Phase 0：Embedding 预聚类（仅 kg 模式）────────────────────────────────
    if mode == "kg":
        eff_threshold = dedup_threshold if dedup_threshold is not None else get_duplicate_threshold()
        eff_model = embedding_model or get_embedding_model()

        try:
            dedup_result = deduplicate_reports_by_embedding(
                reports_all,
                threshold=eff_threshold,
                model=eff_model,
                api_key=get_google_api_key(),
                openai_api_key=get_openai_api_key(),
                base_url=get_base_url(),
            )
            reports = select_representative_reports(reports_all, dedup_result)
            print(
                "[dedup] "
                f"model={dedup_result.model} "
                f"threshold={dedup_result.threshold} "
                f"representatives={len(dedup_result.representatives)}/{dedup_result.total_reports}"
            )
        except Exception as e:
            # If embedding step fails, keep pipeline available with original reports.
            print(f"[dedup] disabled due to error: {type(e).__name__}: {e}")
            dedup_result = None
            reports = reports_all

    # ── Phase 1：实体提取 ──────────────────────────────────────────────────────
    reports = extract_all(reports, app=app, force_refresh=force_refresh, model=model, use_vision=use_vision)
    if use_vision:
        try:
            reports = enrich_reports(
                reports,
                app=app,
                use_vision=use_vision,
                force_refresh=force_refresh,
                model=model,
            )
        except Exception as e:
            print(f"[enricher] disabled due to error: {type(e).__name__}: {e}")

    # ── Phase 2：KG 扩充 + 图匹配分类 ─────────────────────────────────────────
    kg = BugKnowledgeGraph()
    kg.enrich_from_extractions(reports)           # 将 specific_pages/elements 加入图
    classify_all(reports, kg, model=model)                     # 图匹配打分
    llm_fallback_classify(reports, model=model)                # 兜底：对无匹配报告用 LLM 分类
    print(f"[graph]  {kg.summary()}")

    # ── Phase 3：分类后严重性推导 ──────────────────────────────────────────────
    if skip_severity_calculation:
        _assign_default_severity(reports)
        print("[severity] skipped. All reports use neutral severity (priority=0.0).")
    else:
        severities = compute_all_severities(reports, kg)
        for bt, (lbl, score) in sorted(severities.items(), key=lambda x: -x[1][1]):
            count = sum(1 for r in reports if r.get("kg_class", {}).get("bug_type") == bt)
            print(f"  {bt:12s} → {lbl}({score:.2f})  [{count} reports]")

    # ── Phase 4：LLM 排序 / KG 贪心排序 ───────────────────────────────────────
    repeats = given_range or list(range(1, repeat + 1))

    if use_severity_sort:
        subset_sequence = _severity_desc_sequence(reports)

        if dedup_result is not None:
            representative_sequence = _map_subset_to_original(
                subset_sequence,
                dedup_result.representatives,
            )
            final_sequence = expand_ranked_sequence(
                representative_sequence,
                dedup_result,
                total_reports=len(reports_all),
            )
        else:
            final_sequence = subset_sequence

        result_str = _format_result_sequence(final_sequence)
        for i in repeats:
            with open(f"result/{app}-{file_mode}-{i}.txt", encoding="utf-8", mode="w") as f:
                f.write(result_str)
        print("[severity-sort] Done.")
        return

    if mode == "kg-only":
        result_str = _kg_greedy_sort(reports, kg)
        for i in repeats:
            with open(f"result/{app}-{file_mode}-{i}.txt", encoding="utf-8", mode="w") as f:
                f.write(result_str)
        print(f"[kg-only] Done.")
        return

    # "kg" mode：调用 LLM
    prompt = get_kg_prompt(reports, kg)
    for i in repeats:
        print(f"  [LLM] iteration-{i}...")
        llm_result = ask_completion(
            prompt,
            model=model,
            max_token=KG_RANKING_MAX_TOKEN,
        )
        response = llm_result.get("content", "")
        token_p = llm_result.get("prompt_tokens", 0)
        token_r = llm_result.get("completion_tokens", 0)
        with open(f"response/{app}-{file_mode}-{i}.txt", encoding="utf-8", mode="w") as f:
            if response:
                f.write(f"{response}\n")
            f.write(f"Input Tokens: {token_p}\n")
            f.write(f"Output Tokens: {token_r}\n")
            f.write(f"Provider: {llm_result.get('provider', '')}\n")
            f.write(f"Finish Reason: {llm_result.get('finish_reason', '')}\n")
        # 立即解析，写 result
        try:
            if not response.strip():
                raise ValueError("Model returned empty content.")
            subset_sequence = _parse_numbered_sequence(response)
            _validate_subset_sequence(subset_sequence, len(reports))

            # If pre-clustering is active, expand compressed ranking back to all reports.
            if dedup_result is not None:
                representative_sequence = _map_subset_to_original(
                    subset_sequence,
                    dedup_result.representatives,
                )
                final_sequence = expand_ranked_sequence(
                    representative_sequence,
                    dedup_result,
                    total_reports=len(reports_all),
                )
            else:
                final_sequence = subset_sequence

            result = _format_result_sequence(final_sequence)
            with open(f"result/{app}-{file_mode}-{i}.txt", encoding="utf-8", mode="w") as f:
                f.write(result)
        except Exception as e:
            print(f"  [parse] Error: {e}. Please manually fix result/{app}-{file_mode}-{i}.txt")


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
    import argparse

    parser = argparse.ArgumentParser(description="KG-RAG enhanced pipeline")
    parser.add_argument("--app", type=int, default=1, help="app number")
    parser.add_argument("--repeat", type=int, default=5, help="repeat times")
    parser.add_argument("--mode", type=str, default="kg", choices=["kg", "kg-only"], help="pipeline mode")
    parser.add_argument("--model", type=str, default="deepseek-v4-pro", help="LLM model name")
    parser.add_argument("--embedding-model", type=str, default="gemini-embedding-2-preview", help="embedding model name")
    parser.add_argument("--dedup-threshold", type=float, default=0.85, help="dedup threshold (default: config value)")
    parser.add_argument("--skip-severity", default=True, action=argparse.BooleanOptionalAction, help="skip severity calculation")
    parser.add_argument("--use-severity-sort", action="store_true", default=False, help="use severity-only sort instead of LLM")
    parser.add_argument("--force-refresh", action="store_true", default=False, help="force re-extract entities")
    parser.add_argument("--use-vision", action="store_true", default=False, help="use multimodal model with screenshots for entity extraction")
    args = parser.parse_args()

    # File naming: {app}-{mode}-{model}-t{threshold}-{i}.txt  e.g. "1-kg-only-gpt-5.4-t1.0-1.txt"
    file_tag = f"{args.mode}-{args.model}-t{args.dedup_threshold}"

    run_pipeline(
        app=args.app,
        repeat=args.repeat,
        mode=args.mode,
        output_tag=file_tag,
        model=args.model,
        embedding_model=args.embedding_model,
        dedup_threshold=args.dedup_threshold,
        skip_severity_calculation=args.skip_severity,
        use_severity_sort=args.use_severity_sort,
        force_refresh=args.force_refresh,
        use_vision=args.use_vision,
    )
    run_evaluate(app=args.app, repeat=args.repeat, mode=file_tag, print_each=True)
