"""
实体提取模块（Phase 1）

只做关键词提取，不做 bug_type 分类。
对每份报告提取：
  - operations:        触发操作关键词（从预定义列表语义匹配）
  - symptoms:          缺陷症状关键词（从预定义列表语义匹配）
  - specific_pages:    文本中提及的具体页面名称，如"登录页""个人中心"
  - specific_elements: 文本中提及的具体UI元素，如"登录按钮""搜索框"
  - page_type:         对应的抽象页面类型（5选1）
  - module:            所属功能模块（6选1）

结果缓存到 cache/app{n}_entities.json，避免重复调用。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from kg.graph import ALL_MODULES, ALL_OPERATIONS, ALL_PAGES, ALL_SYMPTOMS
from llm import ask_completion

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 提示词
# ──────────────────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """
You are analyzing a crowdsourced bug report. Extract entities WITHOUT classifying the bug type.

Report: {description}

Return a JSON object with exactly these fields:
{{
  "operations":        [list of operations from: {ops}],
  "symptoms":          [list of symptoms from: {syms}],
  "specific_pages":    [specific page/screen names mentioned in the text, e.g. "登录页", "首页"],
  "specific_elements": [specific UI element names mentioned, e.g. "登录按钮", "搜索框", "标题"],
  "page_type":         "<one of: {pages}>",
  "module":            "<one of: {modules}>"
}}

Rules:
- operations/symptoms: use SEMANTIC matching even if exact words differ
  (e.g. "应用退出" → "退出", "显示不全" → "缺失", "黑屏" → "黑屏")
- specific_pages: extract verbatim or near-verbatim page names from the report text
- specific_elements: extract verbatim or near-verbatim element names (buttons, fields, etc.)
- Return ONLY the JSON object, no other text.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def _default_entities() -> Dict[str, Any]:
    return {
        "operations": [],
        "symptoms": [],
        "specific_pages": [],
        "specific_elements": [],
        "page_type": "",
        "module": "",
    }


def _extract_single(description: str, debug: bool = False, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    prompt = _EXTRACT_PROMPT.format(
        description=description,
        ops="、".join(ALL_OPERATIONS),
        syms="、".join(ALL_SYMPTOMS),
        pages="、".join(ALL_PAGES),
        modules="、".join(ALL_MODULES),
    )
    try:
        resp = ask_completion(
            prompt,
            model=model,
            max_token=400,
            temperature=0.0,
        )
        raw = resp.get("content", "")
        if debug:
            print(f"[extractor DEBUG] raw response:\n{raw}\n")
        result = _parse_json(raw)
        # 校验必要字段
        for key in ("operations", "symptoms", "specific_pages", "specific_elements"):
            if not isinstance(result.get(key), list):
                result[key] = []
        return result
    except Exception as e:
        print(f"[extractor ERROR] {type(e).__name__}: {e}")
        return _default_entities()


# ──────────────────────────────────────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────────────────────────────────────

def extract_all(
    reports: List[Dict[str, Any]],
    app: int,
    force_refresh: bool = False,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    对整个应用的所有报告进行实体提取，结果写回报告的 "entities" 字段。

    Args:
        reports:       load_reports() 返回的原始报告列表
        app:           应用编号（用于缓存）
        force_refresh: 忽略缓存，重新提取
        model:         提取用的 LLM 模型名

    Returns:
        原列表，每个元素增加 "entities" 字段
    """
    cache_path = os.path.join(CACHE_DIR, f"app{app}_entities.json")

    # 读取缓存
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache_list = json.load(f)
        idx_map = {item["index"]: item["entities"] for item in cache_list}
        for r in reports:
            r["entities"] = idx_map.get(r["index"], _default_entities())
        print(f"[extractor] Cache loaded: {cache_path} ({len(reports)} reports)")
        return reports

    print(f"[extractor] Extracting entities for app{app} ({len(reports)} reports)...")
    to_cache = []
    for i, r in enumerate(reports):
        entities = _extract_single(r["description"], model=model)
        r["entities"] = entities
        to_cache.append({"index": r["index"], "entities": entities})
        if (i + 1) % 20 == 0 or (i + 1) == len(reports):
            print(f"  [{i + 1}/{len(reports)}]")

    with open(cache_path, encoding="utf-8", mode="w") as f:
        json.dump(to_cache, f, ensure_ascii=False, indent=2)
    print(f"[extractor] Cache saved: {cache_path}")
    return reports
