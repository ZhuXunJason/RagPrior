"""
KG TF-IDF 分类器（Phase 2，修订版）

替换原来无依据的 0.65/0.35 固定系数，改用 IDF 加权图匹配。

理论依据：
  IDF(特征) = log(N / df)
    N  = 缺陷类型总数（10）
    df = 包含该特征的缺陷类型数量

  直觉：
    "崩溃" → 仅属于 崩溃问题         → df=1 → IDF=log(10)=2.30  （高区分度）
    "点击" → 属于 功能/崩溃/错误/跳转 → df=4 → IDF=log(2.5)=0.92 （低区分度）

  评分 = Σ IDF(匹配特征) / Σ IDF(该类型全部关联特征)
  归一化使不同类型间分数可比，无需人工调权。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from kg.graph import ALL_BUG_TYPES, BugKnowledgeGraph


# ──────────────────────────────────────────────────────────────────────────────
# IDF 计算（结果缓存在 kg 对象上，无需重复计算）
# ──────────────────────────────────────────────────────────────────────────────

def compute_idf(kg: BugKnowledgeGraph) -> Dict[str, float]:
    """
    为 KG 中所有 BugSymptom 和 Operation 节点计算 IDF 权重。
    结果缓存到 kg._idf，同一实例只计算一次。
    """
    if hasattr(kg, "_idf"):
        return kg._idf

    n = len(ALL_BUG_TYPES)
    idf: Dict[str, float] = {}

    for node, data in kg.G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype not in ("BugSymptom", "Operation"):
            continue
        # df：有多少个 BugType 通过 associatedWith/TriggeredBy 指向该节点
        df = sum(
            1 for bt in ALL_BUG_TYPES
            if kg.G.has_edge(bt, node)
        )
        idf[node] = math.log(n / max(df, 1))

    kg._idf = idf
    return idf


def classify_report(
    entities: Dict[str, Any],
    kg: BugKnowledgeGraph,
    idf: Dict[str, float],
) -> Dict[str, Any]:
    """
    TF-IDF 加权图匹配分类。

    多个 operation/symptom 全部参与打分，IDF 自动下权区分度低的特征，
    无需人工指定哪个更重要。

    Returns:
        {
          bug_type:   最高分 BugType,
          score:      归一化分数 (0~1),
          evidence:   匹配到的高权重特征（供 LLM 验证用）,
          all_scores: {BugType: score},
          fallback:   所有分均为0时为 True
        }
    """
    rep_syms = set(entities.get("symptoms", []))
    rep_ops  = set(entities.get("operations", []))

    all_scores: Dict[str, float] = {}
    best_evidence: List[str] = []

    for bt in ALL_BUG_TYPES:
        kg_syms = {
            n for n in kg.G.successors(bt)
            if kg.G.nodes[n].get("node_type") == "BugSymptom"
        }
        kg_ops = {
            n for n in kg.G.successors(bt)
            if kg.G.nodes[n].get("node_type") == "Operation"
        }

        matched = (rep_syms & kg_syms) | (rep_ops & kg_ops)
        match_score   = sum(idf.get(f, 0) for f in matched)
        max_possible  = sum(idf.get(f, 0) for f in kg_syms | kg_ops)
        score = match_score / max(max_possible, 1e-9)
        all_scores[bt] = round(score, 4)

    best_bt    = max(all_scores, key=all_scores.__getitem__)
    best_score = all_scores[best_bt]

    # 提取证据：与 best_bt 匹配且 IDF 最高的前3个特征
    kg_feats = {
        n for n in kg.G.successors(best_bt)
        if kg.G.nodes[n].get("node_type") in ("BugSymptom", "Operation")
    }
    matched_feats = (rep_syms | rep_ops) & kg_feats
    best_evidence = sorted(matched_feats, key=lambda f: -idf.get(f, 0))[:3]

    return {
        "bug_type":   best_bt,
        "score":      best_score,
        "evidence":   best_evidence,   # 高权重匹配特征，注入提示词供 LLM 验证
        "all_scores": all_scores,
        "fallback":   best_score < 1e-9,
    }


def classify_all(
    reports: List[Dict[str, Any]],
    kg: BugKnowledgeGraph,
) -> List[Dict[str, Any]]:
    idf = compute_idf(kg)
    fallback = sum(1 for r in reports
                   if classify_report(r.get("entities", {}), kg, idf)["fallback"])
    # 实际分类
    for r in reports:
        r["kg_class"] = classify_report(r.get("entities", {}), kg, idf)

    total = len(reports)
    print(f"[classifier] {total} reports | fallback: {fallback} "
          f"| coverage: {round((total-fallback)/total*100,1)}%")
    return reports


# ──────────────────────────────────────────────────────────────────────────────
# LLM 兜底（仅处理 fallback 报告）
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_PROMPT = """
Classify this bug report into ONE of:
{bug_types}

Report: {description}

Return ONLY the bug type name (exactly as listed), nothing else.
""".strip()


def llm_fallback_classify(reports: List[Dict[str, Any]], client) -> None:
    """对 fallback==True 的报告用 LLM 做一次兜底分类，直接修改 report。"""
    targets = [r for r in reports if r.get("kg_class", {}).get("fallback")]
    if not targets:
        return

    print(f"[classifier] LLM fallback for {len(targets)} reports...")
    bug_types_str = "\n".join(f"- {bt}" for bt in ALL_BUG_TYPES)

    for r in targets:
        prompt = _FALLBACK_PROMPT.format(
            bug_types=bug_types_str,
            description=r["description"],
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            predicted = resp.choices[0].message.content.strip()
            if predicted in ALL_BUG_TYPES:
                r["kg_class"]["bug_type"]  = predicted
                r["kg_class"]["fallback"]  = False
                r["kg_class"]["evidence"]  = []
        except Exception as e:
            print(f"[classifier fallback ERROR] {e}")
