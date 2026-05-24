"""
Graph RAG 增强提示词（修订版）

LLM 的输入现在包含三层 KG 知识：
  ① 每份报告：KG 子图路径 + 分类证据（TF-IDF 推导出的特征） + 结构相似报告
  ② 全局统计：缺陷类型 priority 排序 + 模块分布
  ③ 明确的排序指令，基于 KG 提供的结构信息

这样 LLM 不再仅凭参数记忆推断，而是基于扩容后的 KG 所提供的
真实应用知识（页面、元素、模块路径）来做分类和排序决策。
"""
from __future__ import annotations

from typing import Any, Dict, List

from kg.graph import BugKnowledgeGraph
from kg.rag import build_full_context

_SYSTEM = """
You are a software testing analyst with access to a knowledge graph (KG) about this app.

The KG has been enriched with app-specific pages, elements, and modules extracted from all reports.
Each report below shows its KG-derived context:
  [KG路径]   The actual path in the KG: operations → elements → pages → modules
  [分类证据] The TF-IDF weighted features (symptoms/operations) that led to the bug type classification
  [结构相似] Other reports sharing the same KG nodes (same page/element area) — these are likely in the same defect region
  [视觉证据] Screenshot-derived page state, visible elements, visual defects, and visible error text
""".strip()

_TASK = """
Using the KG context above, prioritize ALL reports to maximize defect discovery efficiency.

Prioritization rules (apply in strict order):
1. DIVERSITY FIRST: Ensure each distinct bug type appears as early as possible.
   - Use [分类证据] and [KG路径] to judge whether two reports truly represent different bugs,
     even if their text descriptions sound similar.
   - Reports with [结构相似] hints are likely in the same defect region — spread them out.
   - Use [视觉证据] to verify page state, visible elements, and visual defects when screenshots are available.

2. SEVERITY SECOND: Among reports that contribute equal diversity value,
   prefer higher priority scores (CRITICAL > HIGH > MEDIUM > LOW).
   The priority score already factors in both KG-defined severity AND how many reports share the same bug.

3. MODULE COVERAGE: Try to cover different modules (see [模块分布]) early.
   Avoid placing reports from the same module consecutively when alternatives exist.

4. KG UNCERTAINTY: Reports marked ⚠LLM-fallback had no strong KG match — treat their
   bug type as approximate. When in doubt about placement, err toward diversity.
   If visual evidence suggests a higher-confidence bug type than text, prefer that signal.

Output ONLY a numbered list, one report per line. No explanations.
Example:
1. Report 5
2. Report 12
3. Report 3
""".strip()

PROMPT_KG = """
{system}

Reports with KG context:
---
{reports_text}
---

{global_summary}

{task}
""".strip()


def get_kg_prompt(
    reports: List[Dict[str, Any]],
    kg: BugKnowledgeGraph,
) -> str:
    """
    构建完整的 Graph RAG 提示词。

    Args:
        reports: 已完成 Phase 1~3 处理的报告列表（含 entities/kg_class/severity）
        kg:      扩容后的 BugKnowledgeGraph 实例
    """
    ctx = build_full_context(reports, kg)
    return PROMPT_KG.format(
        system=_SYSTEM,
        reports_text=ctx["reports_text"],
        global_summary=ctx["global_summary"],
        task=_TASK,
    )
