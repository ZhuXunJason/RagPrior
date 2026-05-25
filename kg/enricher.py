"""
报告图像分析模块

通过多模态 LLM 分析截图，提取视觉缺陷信息。
结果缓存至 cache/app{n}_enriched_vision.json，避免重复 API 调用。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from kg.graph import ALL_BUG_TYPES
from llm import ask_completion

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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


def _image_enrich_single(img_url: str, model: str = "gpt-5.4") -> Optional[Dict[str, Any]]:
    """
    对报告截图进行多模态分析（需要图片 URL 可公开访问）。
    失败时返回 None，调用方应降级处理。
    """
    prompt = _IMAGE_ENRICH_PROMPT.format(bug_types="、".join(ALL_BUG_TYPES))
    try:
        resp = ask_completion(
            prompt,
            model=model,
            max_token=512,
            temperature=0.0,
            image_urls=[img_url],
        )
        raw = resp.get("content", "")
        return _parse_json_response(raw)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────────────────────────────────────

def enrich_reports(
    reports: List[Dict[str, Any]],
    app: int,
    use_vision: bool = False,
    force_refresh: bool = False,
    model: str = "gpt-5.4",
) -> List[Dict[str, Any]]:
    """
    对应用报告进行截图分析并返回富化后的列表。

    Args:
        reports:       load_reports() 返回的原始报告列表
        app:           应用编号（用于缓存文件命名）
        use_vision:    是否启用截图分析
        force_refresh: 忽略缓存，重新提取
        model:         多模态模型名

    Returns:
        每个元素在原始报告字段基础上增加 "kg_meta" 字段，包含 image_meta 视觉分析结果。
    """
    cache_suffix = "_vision" if use_vision else ""
    cache_path = os.path.join(CACHE_DIR, f"app{app}_enriched{cache_suffix}.json")

    # 读取缓存
    if not force_refresh and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        cache_map = {item["index"]: item["kg_meta"] for item in cached}
        for r in reports:
            r["kg_meta"] = cache_map.get(r["index"], _default_kg_meta())
        print(f"[enricher] Loaded cache: {cache_path}")
        return reports

    print(f"[enricher] Enriching {len(reports)} reports for app{app} ...")
    enriched = []
    for i, r in enumerate(reports):
        image_meta = None
        if use_vision and r.get("img_url"):
            image_meta = _image_enrich_single(r["img_url"], model=model)

        r["kg_meta"] = {"text_meta": {}, "image_meta": image_meta}
        enriched.append({"index": r["index"], "kg_meta": r["kg_meta"]})

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(reports)}] done")

    with open(cache_path, encoding="utf-8", mode="w") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"[enricher] Cache saved: {cache_path}")

    return reports


def _default_kg_meta() -> Dict[str, Any]:
    """缓存缺失时的默认值。"""
    return {"text_meta": {}, "image_meta": None}


def get_report_kg_info(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    从富化后的报告中提取关键视觉信息。

    返回: {visual_defect, image_page, error_messages}
    """
    meta = report.get("kg_meta", {})
    image = meta.get("image_meta") or {}
    return {
        "visual_defect":  image.get("visual_defect", ""),
        "image_page":     image.get("page_description", ""),
        "error_messages": image.get("error_messages", []),
    }
