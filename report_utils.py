"""
Shared utilities for report loading, file naming, and APSD/NDCG evaluation.
"""

from typing import Dict, Any, List, Optional, Iterable, Tuple
import os
import re
import numpy as np

try:
    from sklearn.metrics import ndcg_score  # type: ignore
except Exception:
    ndcg_score = None

from config import RESPONSE_DIR, RESULT_DIR
from prompt import load_reports, load_reports_from_file


def resolve_reports(app: int, csv_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load report records from either a custom CSV file or default app file."""
    if csv_file:
        return load_reports_from_file(csv_file)
    return load_reports(app=app)


def resolve_repeats(repeat: int, given_range: Optional[List[int]] = None) -> Iterable[int]:
    """Resolve repetition indices from explicit range or repeat count."""
    if given_range is not None:
        return given_range
    return range(1, repeat + 1)


def get_response_path(app: int, mode: str, iteration: int, model: str) -> str:
    """Build response output file path."""
    return os.path.join(RESPONSE_DIR, f"{app}-{mode}-{iteration}-{model}.txt")


def get_result_path(app: int, mode: str, iteration: int, model: str) -> str:
    """Build parsed result output file path."""
    return os.path.join(RESULT_DIR, f"{app}-{mode}-{iteration}-{model}.txt")


def get_bug_id(report: Dict[str, Any]) -> Optional[Any]:
    """Get bug identifier from report with backward-compatible keys."""
    return report.get("bug_id") or report.get("bug_type")


def collect_bug_types(reports: List[Dict[str, Any]]) -> Tuple[set, int]:
    """Collect unique bug types and return the set and count."""
    bug_ids = {bug_id for bug_id in (get_bug_id(r) for r in reports) if bug_id}
    return bug_ids, len(bug_ids)


def extract_report_num_from_line(line: str) -> Optional[int]:
    """Extract report number from a line like 'Report 7:' or '1. Report 7'."""
    match = re.search(r"Report\s*(\d+)", line)
    if not match:
        return None
    return int(match.group(1))


def extract_sequence_from_lines(lines: List[str]) -> List[int]:
    """Extract ordered report numbers from result lines."""
    sequence: List[int] = []

    for line in lines:
        report_num = extract_report_num_from_line(line)
        if report_num is not None:
            sequence.append(report_num)

    return sequence


def validate_full_permutation(sequence: List[int], total_reports: int) -> bool:
    """Check if the extracted sequence is a valid full permutation of {1..total_reports}."""
    if len(sequence) != total_reports:
        return False
    if set(sequence) != set(range(1, total_reports + 1)):
        return False
    return True


def compute_apfd_from_sequence(sequence: List[int], reports: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute APFD (Average Percentage of Faults Detected).
    """
    total = len(reports)
    if total == 0 or not validate_full_permutation(sequence, total):
        return None

    bug_score = _build_bug_score_map(reports)
    if not bug_score:
        return None

    seen_bug_types = set()
    position_sum = 0
    m = len(bug_score)

    for position, report_num in enumerate(sequence, start=1):
        report_index = report_num - 1
        if report_index < 0 or report_index >= total:
            continue
        bug_id = get_bug_id(reports[report_index])
        if bug_id is None or bug_id in seen_bug_types:
            continue
        seen_bug_types.add(bug_id)
        position_sum += position
        
    if seen_bug_types != set(bug_score.keys()):
        return None

    denom = total * m
    if denom == 0:
        return None

    return 1 - (position_sum / denom) + (1 / (2 * total))


def ensure_report_scores(reports: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Ensure each report has a usable severity score.

    Priority:
    1) Existing report["severity_score"] if provided.
    2) Bug-type frequency as a lightweight fallback score.

    Note:
    The fallback is frequency-weighted rather than a ground-truth severity label.
    Metrics that consume this fallback should be interpreted accordingly.
    """
    bug_freq: Dict[Any, int] = {}
    for report in reports:
        bug_id = get_bug_id(report)
        if bug_id is not None:
            bug_freq[bug_id] = bug_freq.get(bug_id, 0) + 1

    score_map: Dict[int, float] = {}
    for i, report in enumerate(reports, start=1):
        raw_score = report.get("severity_score")
        if raw_score is not None:
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 1.0
        else:
            bug_id = get_bug_id(report)
            score = float(max(1, bug_freq.get(bug_id, 1)))

        report["severity_score"] = score
        score_map[i] = score

    return score_map


def _build_bug_score_map(reports: List[Dict[str, Any]]) -> Dict[Any, float]:
    bug_score: Dict[Any, float] = {}
    ensure_report_scores(reports)
    for report in reports:
        bug_id = get_bug_id(report)
        if bug_id is None:
            continue
        score = float(report.get("severity_score", 1.0))
        if bug_id not in bug_score or score > bug_score[bug_id]:
            bug_score[bug_id] = score
    return bug_score


def compute_apsd_from_sequence(sequence: List[int], reports: List[Dict[str, Any]]) -> Optional[float]:
    """
    APSD = 1 - sum(TF_i * S_i) / (n * sum(S_i)) + 1/(2n)

    TF_i: first occurrence position of each unique bug type
    S_i: severity score for that bug type. When external severity labels are
         unavailable, this project falls back to bug frequency as a proxy.
    n: total number of reports
    """
    total = len(reports)
    if total == 0:
        return None

    bug_score = _build_bug_score_map(reports)
    if not bug_score:
        return None

    seen_bug_types = set()
    weighted_sum = 0.0

    for position, report_num in enumerate(sequence, start=1):
        report_index = report_num - 1
        if report_index < 0 or report_index >= total:
            continue
        bug_id = get_bug_id(reports[report_index])
        if bug_id is None or bug_id in seen_bug_types:
            continue
        seen_bug_types.add(bug_id)
        weighted_sum += position * bug_score.get(bug_id, 1.0)

    if seen_bug_types != set(bug_score.keys()):
        return None

    denom = total * sum(bug_score.values())
    if denom == 0:
        return None

    return 1 - (weighted_sum / denom) + (1 / (2 * total))


def compute_ndcg_from_sequence(sequence: List[int], reports: List[Dict[str, Any]], top_k: Optional[int] = None) -> Optional[float]:
    """Compute NDCG for ranked report sequence using report severity scores as relevance."""
    n = len(reports)
    if n == 0:
        return None

    score_map = ensure_report_scores(reports)
    y_true = np.array([[score_map.get(i, 1.0) for i in range(1, n + 1)]], dtype=float)

    # Higher y_score means higher predicted rank.
    y_pred = np.zeros((1, n), dtype=float)
    for rank, report_num in enumerate(sequence, start=1):
        if 1 <= report_num <= n:
            y_pred[0, report_num - 1] = float(n - rank + 1)

    k = top_k if top_k is not None else n
    k = max(1, min(k, n))

    if ndcg_score is not None:
        return float(ndcg_score(y_true, y_pred, k=k))

    # Fallback if sklearn is unavailable.
    ideal_order = sorted(range(n), key=lambda idx: y_true[0, idx], reverse=True)
    pred_order = sorted(range(n), key=lambda idx: y_pred[0, idx], reverse=True)

    def _dcg(order: List[int]) -> float:
        val = 0.0
        for i, idx in enumerate(order[:k], start=1):
            rel = y_true[0, idx]
            val += rel / np.log2(i + 1)
        return val

    idcg = _dcg(ideal_order)
    if idcg == 0:
        return None
    return float(_dcg(pred_order) / idcg)
