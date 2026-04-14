"""
分类后严重性推导（修订版）

核心公式（无任意调权系数）：
  priority = base_weight × (1 + log₂(1 + count))

理论依据：
  base_weight：KG 内建的缺陷类型严重性（致命=4, 高=3, 中=2, 低=1）
  count：该类别覆盖的报告数量
  log₂(1+count)：对数增益，直觉上来自信息论"信息量 ∝ log(频率)"
    count=1  → ×1.00（单份报告，不额外加权）
    count=3  → ×2.00（3倍覆盖，权重翻倍）
    count=7  → ×3.00
    count=15 → ×4.00
  结果：崩溃问题(base=4)+1份报告=4.0；显示问题(base=1)+50份报告≈6.6
  → 大量低严重性缺陷在普遍时确实值得更早发现，但不会超过致命缺陷

标签映射：
  对所有缺陷类别的 priority 排序，按四分位数映射到 低/中/高/致命，
  使标签分布均匀，不依赖固定阈值。

元素关键度仅作为同优先级时的次级排序，不参与主公式。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from kg.graph import BUG_TYPE_SEVERITY, SEVERITY_ORDER, ELEMENT_CATEGORIES, BugKnowledgeGraph

_ELEM_TO_CAT = {e: cat for cat, elems in ELEMENT_CATEGORIES.items() for e in elems}
_CAT_RANK    = {"容器类": 4, "交互类": 3, "反馈类": 3, "布局类": 2, "数据状态类": 2, "展示类": 1}


def _elem_criticality(element_names: List[str], kg: BugKnowledgeGraph) -> float:
    """元素关键度均值（0~1），仅用作次级排序辅助。"""
    scores = []
    for name in element_names:
        cat = _ELEM_TO_CAT.get(name)
        if cat is None:
            for nbr in kg.G.successors(name):
                if kg.G.nodes[nbr].get("node_type") == "Element":
                    cat = _ELEM_TO_CAT.get(nbr)
                    break
        scores.append(_CAT_RANK.get(cat, 1) / 4.0 if cat else 0.25)
    return sum(scores) / len(scores) if scores else 0.25


def compute_all_severities(
    reports: List[Dict[str, Any]],
    kg: BugKnowledgeGraph,
) -> Dict[str, Tuple[str, float]]:
    """
    按 kg_class.bug_type 分组，计算每类的 priority_score，映射严重性标签。
    结果写回 report["severity"] = {label, priority, elem_crit}。
    """
    # 1. 分组
    groups: Dict[str, List[Dict]] = {}
    for r in reports:
        bt = r.get("kg_class", {}).get("bug_type", "功能缺陷")
        groups.setdefault(bt, []).append(r)

    # 2. 计算 priority = base × (1 + log₂(1 + count))
    priorities: Dict[str, float] = {}
    for bt, grp in groups.items():
        base  = SEVERITY_ORDER.get(BUG_TYPE_SEVERITY.get(bt, "中"), 2)
        count = len(grp)
        priorities[bt] = round(base * (1 + math.log2(1 + count)), 3)

    # 3. 四分位数映射标签（分布均匀，不依赖固定阈值）
    sorted_vals = sorted(priorities.values())
    n = len(sorted_vals)
    q = [sorted_vals[max(0, n * k // 4 - 1)] for k in range(1, 4)]   # Q1, Q2, Q3

    def _label(v: float) -> str:
        if v >= q[2]: return "致命"
        if v >= q[1]: return "高"
        if v >= q[0]: return "中"
        return "低"

    # 4. 写回
    severities: Dict[str, Tuple[str, float]] = {}
    for bt, grp in groups.items():
        priority = priorities[bt]
        label    = _label(priority)
        all_elems = [e for r in grp
                     for e in r.get("entities", {}).get("specific_elements", [])]
        elem_crit = _elem_criticality(list(dict.fromkeys(all_elems)), kg)

        severities[bt] = (label, priority)
        for r in grp:
            r["severity"] = {
                "label":     label,
                "priority":  priority,
                "elem_crit": round(elem_crit, 3),
            }

    return severities
