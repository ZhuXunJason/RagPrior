"""
报告预处理与实体提取模块

对每份众测报告进行两阶段分析：
  1. 文本分析：提取 BugType / BugSymptom / Operation / Page / Module / Element / Severity
  2. 图像分析（可选）：通过 GPT-4o vision 分析截图，提取视觉缺陷信息

结果缓存至 cache/app{n}_enriched.json，避免重复 API 调用。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from config import create_openai_client

from kg.graph import (
    ALL_BUG_TYPES, ALL_MODULES, ALL_OPERATIONS, ALL_PAGES, ALL_SYMPTOMS,
    ELEMENT_CATEGORIES,
)

# 所有元素名（展平）
_ALL_ELEMENTS: List[str] = [e for elems in ELEMENT_CATEGORIES.values() for e in elems]

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_CLIENT = create_openai_client()

# ──────────────────────────────────────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────────────────────────────────────

_TEXT_ENRICH_PROMPT = """
You are an expert software testing analyst. Analyze the following crowdsourced bug report and extract structured metadata.

Report description: {description}

Return a JSON object with exactly these fields (use Chinese values from the provided lists):
{{
  "bug_type": "<one of: {bug_types}>",
  "bug_symptoms": ["<from: {symptoms}>", ...],
  "operations": ["<from: {operations}>", ...],
  "element_types": ["<from: {elements}>", ...],
  "page_type": "<one of: {pages}>",
  "module": "<one of: {modules}>",
  "severity": "<one of: 低/中/高/致命>",
  "confidence": <0.0-1.0>
}}

Rules:
- bug_symptoms: list up to 3 most relevant symptoms
- operations: list up to 3 most relevant triggering operations
- element_types: list up to 3 most relevant UI elements
- severity: infer from the bug's impact (致命=crash/data loss, 高=core feature broken, 中=degraded, 低=cosmetic)
- confidence: how confident you are in the classification (0-1)
- Return ONLY the JSON object, no other text.
""".strip()

_IMAGE_ENRICH_PROMPT = """
Analyze this screenshot from a mobile app bug report.

Return a JSON object with exactly these fields:
{{
  "page_description": "<brief description of the page shown>",
  "visible_elements": ["<UI elements visible>", ...],
  "visual_defect": "<description of the visible defect>",
  "error_messages": ["<any error text visible>"],
  "suggested_bug_type": "<one of: {bug_types}>",
  "severity_hint": "<one of: 低/中/高/致命>"
}}

Return ONLY the JSON object, no other text.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON（兼容 Markdown 代码块包裹的情况）。"""
    # 去除 markdown 代码块
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 {...} 块
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _text_enrich_single(description: str) -> Dict[str, Any]:
    """对单份报告的文本描述进行 LLM 实体提取。"""
    prompt = _TEXT_ENRICH_PROMPT.format(
        description=description,
        bug_types="、".join(ALL_BUG_TYPES),
        symptoms="、".join(ALL_SYMPTOMS[:20]) + "等",
        operations="、".join(ALL_OPERATIONS[:20]) + "等",
        elements="、".join(_ALL_ELEMENTS[:20]) + "等",
        pages="、".join(ALL_PAGES),
        modules="、".join(ALL_MODULES),
    )
    response = _CLIENT.chat.completions.create(
        model="gpt-4o-mini",   # 成本低、速度快；可换成 gpt-4o 获得更好的效果
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content
    result = _parse_json_response(raw)
    if result is None:
        # 解析失败时返回默认值
        result = {
            "bug_type": "功能缺陷",
            "bug_symptoms": [],
            "operations": [],
            "element_types": [],
            "page_type": "数据展示类页面",
            "module": "系统与设置模块",
            "severity": "中",
            "confidence": 0.0,
        }
    return result


def _image_enrich_single(img_url: str) -> Optional[Dict[str, Any]]:
    """
    对报告截图进行 GPT-4o vision 分析（需要图片 URL 可公开访问）。
    失败时返回 None，调用方应降级处理。
    """
    prompt = _IMAGE_ENRICH_PROMPT.format(
        bug_types="、".join(ALL_BUG_TYPES)
    )
    try:
        response = _CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url, "detail": "low"}},
                ],
            }],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        return _parse_json_response(raw)
    except Exception as e:
        # 图片不可访问、API 错误等均静默失败
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────────────────────────────────────

def enrich_reports(
    reports: List[Dict[str, Any]],
    app: int,
    use_vision: bool = False,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    对一个应用的所有报告进行实体提取并返回富化后的列表。

    Args:
        reports:       load_reports() 返回的原始报告列表
        app:           应用编号（用于缓存文件命名）
        use_vision:    是否启用截图分析（需要 GPT-4o，成本较高）
        force_refresh: 忽略缓存，重新提取

    Returns:
        每个元素在原始报告字段基础上增加 "kg_meta" 字段，包含：
          text_meta:   文本实体提取结果
          image_meta:  图像分析结果（仅 use_vision=True 时有值）
    """
    cache_suffix = "_vision" if use_vision else ""
    cache_path = os.path.join(CACHE_DIR, f"app{app}_enriched{cache_suffix}.json")

    # 读取缓存
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        # 按 index 索引
        cache_map = {item["index"]: item["kg_meta"] for item in cached}
        for r in reports:
            r["kg_meta"] = cache_map.get(r["index"], _default_kg_meta())
        print(f"[enricher] Loaded cache: {cache_path}")
        return reports

    print(f"[enricher] Enriching {len(reports)} reports for app{app} ...")
    enriched = []
    for i, r in enumerate(reports):
        text_meta = _text_enrich_single(r["description"])
        image_meta = None
        if use_vision and r.get("img_url"):
            image_meta = _image_enrich_single(r["img_url"])
            # 如果图像分析给出了更高置信度的 bug_type，用它覆盖文本结果
            if image_meta and image_meta.get("suggested_bug_type") in ALL_BUG_TYPES:
                if text_meta.get("confidence", 0) < 0.7:
                    text_meta["bug_type"] = image_meta["suggested_bug_type"]

        r["kg_meta"] = {
            "text_meta": text_meta,
            "image_meta": image_meta,
        }
        enriched.append({"index": r["index"], "kg_meta": r["kg_meta"]})

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(reports)}] done")

    # 写入缓存
    with open(cache_path, encoding="utf-8", mode="w") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"[enricher] Cache saved: {cache_path}")

    return reports


def _default_kg_meta() -> Dict[str, Any]:
    """缓存缺失时的默认值。"""
    return {
        "text_meta": {
            "bug_type": "功能缺陷",
            "bug_symptoms": [],
            "operations": [],
            "element_types": [],
            "page_type": "数据展示类页面",
            "module": "系统与设置模块",
            "severity": "中",
            "confidence": 0.0,
        },
        "image_meta": None,
    }


def get_report_kg_info(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    从富化后的报告中提取关键 KG 字段，方便其他模块使用。

    返回: {bug_type, severity, module, page_type, element_types, image_visual_defect}
    """
    meta = report.get("kg_meta", {})
    text = meta.get("text_meta", {})
    image = meta.get("image_meta") or {}

    return {
        "bug_type":       text.get("bug_type", "功能缺陷"),
        "severity":       text.get("severity", "中"),
        "module":         text.get("module", ""),
        "page_type":      text.get("page_type", ""),
        "element_types":  text.get("element_types", []),
        "visual_defect":  image.get("visual_defect", ""),
        "image_page":     image.get("page_description", ""),
        "error_messages": image.get("error_messages", []),
    }
