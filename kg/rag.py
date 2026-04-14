"""
Graph RAG 上下文构建模块（全面修订）

核心思路：
  KG 扩容后，每份报告在图中有 1~2 跳的结构化邻域。
  本模块负责从 KG 中"检索"与每份报告相关的知识，并转化为
  LLM 可以直接使用的自然语言上下文——这是真正的 Graph RAG。

LLM 得到的信息分三层：
  ① 个体层：每份报告的 KG 子图路径（操作→元素→页面→模块）
             + TF-IDF 分类证据（哪些特征导致了该分类）
             + 结构相似报告（共享同一页面/元素的其他报告编号）
  ② 群体层：报告间的 KG 聚类（按页面分组，帮助 LLM 判断多样性）
  ③ 全局层：模块/缺陷类型的覆盖分布 + priority 排序
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from kg.graph import BugKnowledgeGraph


# ──────────────────────────────────────────────────────────────────────────────
# ① 个体层：每份报告的 KG 上下文行
# ──────────────────────────────────────────────────────────────────────────────

def _format_report_line(
    num: int,
    report: Dict[str, Any],
    kg: BugKnowledgeGraph,
    similar_nums: List[int],
) -> str:
    """
    生成单份报告的带 KG 上下文的多行描述。

    格式：
      Report N - <description>
        [KG路径] 操作:点击 | 症状:崩溃 | 页面:登录页→用户与账号模块 | 元素:登录按钮
        [分类证据] 症状[崩溃,退出] → 崩溃问题 (score:0.85, priority:6.93/CRITICAL)
        [结构相似] #5,#12 共享同一页面/元素（可能属于同一缺陷区域）
    """
    desc    = report.get("description", "")
    klass   = report.get("kg_class", {})
    sev     = report.get("severity", {})
    ents    = report.get("entities", {})
    idx     = report.get("index", "")

    bug_type = klass.get("bug_type", "—")
    score    = klass.get("score", 0.0)
    evidence = klass.get("evidence", [])
    fallback = klass.get("fallback", False)
    priority = sev.get("priority", 0.0)
    label    = sev.get("label", "中")

    _SEV_EN  = {"低": "LOW", "中": "MEDIUM", "高": "HIGH", "致命": "CRITICAL"}

    # 子图路径（Graph RAG 核心）
    kg_path = kg.get_report_subgraph_text(idx)

    # 分类证据字符串
    ev_str = f"症状/操作[{','.join(evidence)}] → " if evidence else ""
    flag   = " ⚠LLM-fallback" if fallback else ""
    cls_str = (f"{ev_str}{bug_type} "
               f"(score:{score:.2f}, priority:{priority}/{_SEV_EN.get(label,'?')}{flag})")

    lines = [f"Report {num} - {desc}"]
    if kg_path:
        lines.append(f"  [KG路径] {kg_path}")
    lines.append(f"  [分类证据] {cls_str}")
    if similar_nums:
        sim_str = ",".join(f"#{n}" for n in similar_nums[:4])
        lines.append(f"  [结构相似] {sim_str} 共享同一页面/元素（同一缺陷区域，注意多样性）")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# ② 群体层：基于 KG 的报告聚类
# ──────────────────────────────────────────────────────────────────────────────

def build_page_clusters(
    reports: List[Dict[str, Any]],
    kg: BugKnowledgeGraph,
) -> Dict[int, List[int]]:
    """
    基于 KG 中共享的页面节点对报告进行聚类。

    两份报告共享同一 AppPage/Page 节点 → 它们在相同页面区域，
    大概率涉及相同 UI 功能，LLM 应避免连续从同一聚类中取报告。

    Returns:
        {report_1based_num: [similar_report_nums...]}
    """
    # page_node → [report_1based_num]
    page_to_reports: Dict[str, List[int]] = defaultdict(list)

    for i, r in enumerate(reports):
        idx      = r.get("index", "")
        pages    = kg.get_shared_pages(idx)
        if pages:
            for pg in pages:
                page_to_reports[pg].append(i + 1)
        else:
            # 无具体页面：按 page_type + module 归组
            ents     = r.get("entities", {})
            pt, mod  = ents.get("page_type", ""), ents.get("module", "")
            key      = f"{pt}/{mod}"
            if key != "/":
                page_to_reports[key].append(i + 1)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for _page, nums in page_to_reports.items():
        if len(nums) < 2:
            continue
        for n in nums:
            clusters[n].extend(x for x in nums if x != n)

    # 去重，最多保留 4 个
    return {n: list(dict.fromkeys(lst))[:4] for n, lst in clusters.items()}


# ──────────────────────────────────────────────────────────────────────────────
# ③ 全局层：分布统计
# ──────────────────────────────────────────────────────────────────────────────

def build_global_kg_summary(
    reports: List[Dict[str, Any]],
) -> str:
    """
    生成 KG 全局视图字符串，让 LLM 了解整个报告集的结构分布。

    包含：
      - 缺陷类型组（按 priority 降序）
      - 模块分布
    """
    type_groups: Dict[str, Dict] = {}
    mod_counter: Dict[str, int]  = defaultdict(int)

    for r in reports:
        bt       = r.get("kg_class", {}).get("bug_type", "—")
        priority = r.get("severity", {}).get("priority", 0.0)
        label    = r.get("severity", {}).get("label", "中")
        mod      = r.get("entities", {}).get("module", "")

        if bt not in type_groups:
            type_groups[bt] = {"count": 0, "priority": priority, "label": label}
        type_groups[bt]["count"] += 1

        if mod:
            mod_counter[mod] += 1

    _SEV_EN = {"低": "LOW", "中": "MEDIUM", "高": "HIGH", "致命": "CRITICAL"}

    # 缺陷类型组，按 priority 降序
    sorted_types = sorted(type_groups.items(), key=lambda x: -x[1]["priority"])
    type_lines = [
        f"{bt}({info['count']}份, priority:{info['priority']}/{_SEV_EN.get(info['label'],'?')})"
        for bt, info in sorted_types
    ]

    mod_lines = [f"{m}({c}份)" for m, c in sorted(mod_counter.items(), key=lambda x: -x[1])]

    lines = []
    if type_lines:
        lines.append("[缺陷类型组(按priority降序)] " + " | ".join(type_lines))
    if mod_lines:
        lines.append("[模块分布] " + " | ".join(mod_lines))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 公共接口：组装完整上下文
# ──────────────────────────────────────────────────────────────────────────────

def build_full_context(
    reports: List[Dict[str, Any]],
    kg: BugKnowledgeGraph,
) -> Dict[str, Any]:
    """
    一次性构建所有层级的 KG 上下文，返回给 prompt_kg.py 使用。

    Returns:
        {
          reports_text:   带 KG 上下文的报告列表字符串,
          global_summary: 全局分布统计字符串,
        }
    """
    clusters = build_page_clusters(reports, kg)

    report_lines = []
    for i, r in enumerate(reports):
        similar = clusters.get(i + 1, [])
        report_lines.append(_format_report_line(i + 1, r, kg, similar))

    return {
        "reports_text":   "\n\n".join(report_lines),
        "global_summary": build_global_kg_summary(reports),
    }
