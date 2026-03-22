"""
Offline KG service backed by local KG_desc.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re


@dataclass
class KGEntity:
    name: str
    surface: str
    aliases: List[str]
    markers: List[str]


@dataclass
class OfflineKG:
    bug_symptoms: List[str]
    operations: List[str]
    severities: List[str]
    elements: List[KGEntity]
    pages: List[KGEntity]
    op_to_elements: Dict[str, List[str]]
    op_to_pages: Dict[str, List[str]]
    op_to_bugs: Dict[str, List[str]]


DEFAULT_PAGE_MARKERS = ["页面", "界面", "页", "目录", "栏目", "中心"]
DEFAULT_ELEMENT_MARKERS = [
    "按钮",
    "图标",
    "入口",
    "菜单",
    "菜单栏",
    "输入框",
    "列表",
    "弹窗",
    "对话框",
    "标签",
    "卡片",
    "开关",
    "复选框",
    "文本",
    "图片",
    "标记",
    "提示框",
    "占位符",
    "键",
    "提示",
    "状态",
]

TAIL_POSITION_WEIGHT_BASE = 0.6
TAIL_POSITION_WEIGHT_SCALE = 0.8


def _parse_section(lines: List[str], start_index: int) -> Tuple[List[str], int]:
    items: List[str] = []
    index = start_index + 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("### "):
            break
        if line.startswith("- "):
            item = line[2:].strip()
        else:
            item = line.strip()
        if item:
            items.append(item)
        index += 1
    return items, index


def _parse_relation_section(lines: List[str], start_index: int) -> Tuple[Dict[str, List[str]], int]:
    relations: Dict[str, List[str]] = {}
    index = start_index + 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("### "):
            break
        if not line.startswith("- ") or "→" not in line:
            index += 1
            continue
        left, right = line[2:].split("→", 1)
        head = left.strip()
        values = [item.strip() for item in right.split("、") if item.strip()]
        if head and values:
            relations[head] = values
        index += 1
    return relations, index


def _split_csv_like(value: str) -> List[str]:
    parts = re.split(r"[、,，]", str(value or ""))
    return [item.strip() for item in parts if item.strip()]


def _derive_surface(name: str, markers: List[str]) -> str:
    normalized = str(name or "").strip()
    for marker in sorted(markers, key=len, reverse=True):
        if normalized.endswith(marker) and len(normalized) > len(marker):
            candidate = normalized[: -len(marker)].strip()
            if len(candidate) >= 2:
                return candidate
    return normalized


def _parse_entity_line(line: str, markers: List[str]) -> KGEntity | None:
    raw = line.strip()
    if not raw:
        return None
    if raw.startswith("- "):
        raw = raw[2:].strip()
    if not raw or raw.startswith("markers:"):
        return None

    if raw.startswith("name:"):
        fields: Dict[str, str] = {}
        for part in raw.split("|"):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        name = fields.get("name", "").strip()
        if not name:
            return None
        surface = fields.get("surface", "").strip() or _derive_surface(name, markers)
        aliases = _split_csv_like(fields.get("aliases", ""))
        return KGEntity(name=name, surface=surface, aliases=aliases, markers=list(markers))

    name = raw.strip()
    if not name:
        return None
    return KGEntity(name=name, surface=_derive_surface(name, markers), aliases=[], markers=list(markers))


def _parse_entity_section(
    lines: List[str],
    start_index: int,
    default_markers: List[str],
) -> Tuple[List[KGEntity], List[str], int]:
    markers = list(default_markers)
    items: List[KGEntity] = []
    index = start_index + 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("### "):
            break
        if line.startswith("- markers:"):
            markers = _split_csv_like(line.split(":", 1)[1])
            index += 1
            continue
        entity = _parse_entity_line(line, markers)
        if entity is not None:
            items.append(entity)
        index += 1
    return items, markers, index


def load_offline_kg(desc_path: str | Path) -> OfflineKG:
    path = Path(desc_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: Dict[str, List[str]] = {
        "bug_symptoms": [],
        "operations": [],
        "severities": [],
        "elements": [],
        "pages": [],
        "element_markers": list(DEFAULT_ELEMENT_MARKERS),
        "page_markers": list(DEFAULT_PAGE_MARKERS),
        "op_to_elements": {},
        "op_to_pages": {},
        "op_to_bugs": {},
    }

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("### BugSymptom"):
            sections["bug_symptoms"], index = _parse_section(lines, index)
            continue
        if line.startswith("### Operation"):
            sections["operations"], index = _parse_section(lines, index)
            continue
        if line.startswith("### Severity"):
            sections["severities"], index = _parse_section(lines, index)
            continue
        if line.startswith("### Element"):
            sections["elements"], sections["element_markers"], index = _parse_entity_section(lines, index, DEFAULT_ELEMENT_MARKERS)
            continue
        if line.startswith("### Page"):
            sections["pages"], sections["page_markers"], index = _parse_entity_section(lines, index, DEFAULT_PAGE_MARKERS)
            continue
        if line.startswith("### OperatesOnElement"):
            sections["op_to_elements"], index = _parse_relation_section(lines, index)
            continue
        if line.startswith("### NavigatesToPage"):
            sections["op_to_pages"], index = _parse_relation_section(lines, index)
            continue
        if line.startswith("### TriggersBugSymptom"):
            sections["op_to_bugs"], index = _parse_relation_section(lines, index)
            continue
        index += 1

    return OfflineKG(
        bug_symptoms=sections["bug_symptoms"],
        operations=sections["operations"],
        severities=sections["severities"],
        elements=sections["elements"],
        pages=sections["pages"],
        op_to_elements=sections["op_to_elements"],
        op_to_pages=sections["op_to_pages"],
        op_to_bugs=sections["op_to_bugs"],
    )


class OfflineKGService:
    supports_priority_evidence = False

    def __init__(self, desc_path: str | Path):
        self.desc_path = str(desc_path)
        kg = load_offline_kg(desc_path)
        self.operations = kg.operations
        self.elements = kg.elements
        self.pages = kg.pages
        self.bug_symptoms = kg.bug_symptoms
        self.severities = kg.severities
        self.op_to_elements = kg.op_to_elements
        self.op_to_pages = kg.op_to_pages
        self.op_to_bugs = kg.op_to_bugs

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _strip_suffixes(text: str) -> str:
        stripped = text
        for suffix in ("页面", "页面中", "界面", "页面", "页", "按钮", "图标", "列表", "输入框", "提示框", "提示", "状态"):
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[: -len(suffix)]
                break
        return stripped

    @staticmethod
    def _char_overlap_ratio(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_chars = set(left)
        right_chars = set(right)
        overlap = len(left_chars & right_chars)
        base = max(len(left_chars), len(right_chars), 1)
        return overlap / float(base)

    @staticmethod
    def _latest_match_ratio(query: str, needle: str) -> float | None:
        if not query or not needle:
            return None
        idx = query.rfind(needle)
        if idx < 0:
            return None
        denominator = max(len(query) - len(needle), 1)
        return min(max(idx / float(denominator), 0.0), 1.0)

    @staticmethod
    def _tail_position_weight(position_ratio: float | None) -> float:
        if position_ratio is None:
            return 1.0
        return TAIL_POSITION_WEIGHT_BASE + TAIL_POSITION_WEIGHT_SCALE * position_ratio

    def _score_entity(
        self,
        query: str,
        entity: str,
        *,
        tail_bias: bool = False,
    ) -> Tuple[float, float, float, str]:
        query_norm = self._normalize_text(query)
        entity_norm = self._normalize_text(entity)
        if not query_norm or not entity_norm:
            return 0.0, 0.0, 1.0, entity_norm

        score = 0.0
        position_ratio: float | None = None
        matched_form = entity_norm
        if entity_norm in query_norm:
            score += 3.0
            position_ratio = self._latest_match_ratio(query_norm, entity_norm)
        if query_norm in entity_norm:
            score += 1.0

        query_base = self._strip_suffixes(query_norm)
        entity_base = self._strip_suffixes(entity_norm)
        if entity_base and entity_base in query_norm:
            score += 1.5
            base_ratio = self._latest_match_ratio(query_norm, entity_base)
            if base_ratio is not None and (position_ratio is None or base_ratio >= position_ratio):
                position_ratio = base_ratio
                matched_form = entity_base
        if query_base and query_base in entity_norm:
            score += 0.8

        score += self._char_overlap_ratio(query_norm, entity_norm)
        position_weight = 1.0
        if tail_bias and position_ratio is not None:
            position_weight = self._tail_position_weight(position_ratio)
            score *= position_weight
        return score, float(position_ratio or 0.0), position_weight, matched_form

    def _score_form(
        self,
        query: str,
        form: str,
        surface: str,
        *,
        tail_bias: bool = False,
    ) -> Tuple[float, float, float]:
        query_norm = self._normalize_text(query)
        form_norm = self._normalize_text(form)
        surface_norm = self._normalize_text(surface)
        if not query_norm or not form_norm:
            return 0.0, 0.0, 1.0

        score = 0.0
        position_ratio: float | None = None
        if form_norm in query_norm:
            score += 3.0
            position_ratio = self._latest_match_ratio(query_norm, form_norm)
        if surface_norm and surface_norm in query_norm:
            score += 1.5
            surface_ratio = self._latest_match_ratio(query_norm, surface_norm)
            if surface_ratio is not None and (position_ratio is None or surface_ratio >= position_ratio):
                position_ratio = surface_ratio
        score += self._char_overlap_ratio(query_norm, form_norm)
        position_weight = 1.0
        if tail_bias and position_ratio is not None:
            position_weight = self._tail_position_weight(position_ratio)
            score *= position_weight
        return score, float(position_ratio or 0.0), position_weight

    @staticmethod
    def _entity_forms(entity: KGEntity) -> List[Tuple[str, str | None]]:
        forms: List[Tuple[str, str | None]] = []
        seen = set()

        def add(form: str, marker: str | None = None) -> None:
            normalized = " ".join(str(form or "").split()).strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            forms.append((normalized, marker))

        add(entity.name, None)
        add(entity.surface, None)
        for alias in entity.aliases:
            add(alias, None)
        for marker in entity.markers:
            add(f"{entity.surface}{marker}", marker)
        return forms

    def _rank_structured_entities(
        self,
        query: str,
        entities: List[KGEntity],
        top_k: int,
        prefix: str,
        operation_candidates: List[Dict[str, Any]] | None = None,
        relation_map: Dict[str, List[str]] | None = None,
    ) -> List[Dict[str, Any]]:
        operation_candidates = operation_candidates or []
        relation_map = relation_map or {}
        ranked: List[Dict[str, Any]] = []

        for entity in entities:
            best_form = entity.name
            best_marker = None
            text_score = 0.0
            best_position_ratio = 0.0
            best_position_weight = 1.0
            for form, marker in self._entity_forms(entity):
                current, position_ratio, position_weight = self._score_form(
                    query,
                    form,
                    entity.surface,
                    tail_bias=True,
                )
                if current > text_score:
                    text_score = current
                    best_form = form
                    best_marker = marker
                    best_position_ratio = position_ratio
                    best_position_weight = position_weight

            relation_bonus = 0.0
            relation_ops: List[str] = []
            for idx, op_item in enumerate(operation_candidates):
                op_name = str(op_item.get("name", "")).strip()
                if entity.name not in relation_map.get(op_name, []):
                    continue
                relation_ops.append(op_name)
                relation_bonus = max(relation_bonus, 0.8 if idx == 0 else 0.4)

            final_score = text_score + relation_bonus
            if text_score < 0.6 and final_score < 1.0:
                continue

            ranked.append(
                {
                    "name": entity.name,
                    "score": final_score,
                    "source": "kg",
                    "evidence": {
                        "kg": True,
                        "relation": bool(relation_ops),
                        "fallback": False,
                        "matched_surface": entity.surface,
                        "matched_form": best_form,
                        "matched_marker": best_marker or "",
                        "relation_bonus": relation_bonus,
                        "relation_ops": relation_ops,
                        "text_score": text_score,
                        "position_ratio": best_position_ratio,
                        "position_weight": best_position_weight,
                    },
                }
            )

        ranked.sort(key=lambda item: (float(item["score"]), len(item["name"])), reverse=True)
        return [
            {
                "id": f"{prefix}{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": item["source"],
                "evidence": dict(item.get("evidence", {})),
            }
            for idx, item in enumerate(ranked[:top_k], start=1)
        ]

    def _rank_entities(
        self,
        query: str,
        entities: List[str],
        top_k: int,
        prefix: str,
        *,
        tail_bias: bool = False,
    ) -> List[Dict[str, Any]]:
        scored = []
        for entity in entities:
            score, position_ratio, position_weight, matched_form = self._score_entity(
                query,
                entity,
                tail_bias=tail_bias,
            )
            if score < 0.45:
                continue
            evidence = {
                "kg": True,
                "relation": False,
                "fallback": False,
            }
            if tail_bias:
                evidence.update(
                    {
                        "matched_form": matched_form,
                        "position_ratio": position_ratio,
                        "position_weight": position_weight,
                    }
                )
            scored.append(
                {
                    "name": entity,
                    "score": score,
                    "evidence": evidence,
                }
            )
        scored.sort(key=lambda item: (float(item["score"]), len(item["name"])), reverse=True)
        return [
            {
                "id": f"{prefix}{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": "kg",
                "evidence": dict(item.get("evidence", {})),
            }
            for idx, item in enumerate(scored[:top_k], start=1)
        ]

    @staticmethod
    def _max_candidate_score(candidates: List[Dict[str, Any]]) -> float:
        if not candidates:
            return 0.0
        return max(float(item.get("score", 0.0)) for item in candidates)

    @staticmethod
    def _merge_ranked_entities(
        direct: List[Dict[str, Any]],
        related: List[Dict[str, Any]],
        top_k: int,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for bucket, boost in ((direct, 0.0), (related, 0.0)):
            for item in bucket:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                entry = merged.setdefault(name, {"name": name, "score": 0.0, "source": set()})
                entry["score"] += float(item.get("score", 0.0)) + boost
                source = str(item.get("source", "")).strip()
                if source:
                    entry["source"].add(source)
                evidence = item.get("evidence", {}) or {}
                merged_evidence = entry.setdefault("evidence", {})
                for key, value in evidence.items():
                    if isinstance(value, bool):
                        merged_evidence[key] = bool(merged_evidence.get(key)) or value
                    else:
                        merged_evidence[key] = value
        ranked = sorted(
            merged.values(),
            key=lambda item: (float(item["score"]), len(item["name"])),
            reverse=True,
        )[:top_k]
        return [
            {
                "id": f"{prefix}{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": ",".join(sorted(item["source"])),
                "evidence": dict(item.get("evidence", {})),
            }
            for idx, item in enumerate(ranked, start=1)
        ]

    def _rank_related_entities(
        self,
        query: str,
        operation_candidates: List[Dict[str, Any]],
        relation_map: Dict[str, List[str]],
        prefix: str,
        top_k: int,
        *,
        tail_bias: bool = False,
    ) -> List[Dict[str, Any]]:
        related_scores: Dict[str, Dict[str, Any]] = {}
        for op_item in operation_candidates:
            op_name = str(op_item.get("name", "")).strip()
            op_score = float(op_item.get("score", 0.0))
            for related_name in relation_map.get(op_name, []):
                direct_score, position_ratio, position_weight, matched_form = self._score_entity(
                    query,
                    related_name,
                    tail_bias=tail_bias,
                )
                total_score = 0.7 * op_score + direct_score
                if total_score < 0.6:
                    continue
                entry = related_scores.setdefault(
                    related_name,
                    {
                        "name": related_name,
                        "score": 0.0,
                        "source": set(),
                        "evidence": {
                            "kg": True,
                            "relation": True,
                            "fallback": False,
                            "matched_form": matched_form,
                            "position_ratio": position_ratio,
                            "position_weight": position_weight,
                        },
                    },
                )
                entry["score"] += total_score
                entry["source"].add("kg")
                if tail_bias and position_ratio >= float(entry["evidence"].get("position_ratio", 0.0)):
                    entry["evidence"]["matched_form"] = matched_form
                    entry["evidence"]["position_ratio"] = position_ratio
                    entry["evidence"]["position_weight"] = position_weight
        ranked = sorted(
            related_scores.values(),
            key=lambda item: (float(item["score"]), len(item["name"])),
            reverse=True,
        )[:top_k]
        return [
            {
                "id": f"{prefix}{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": ",".join(sorted(item["source"])),
                "evidence": dict(item.get("evidence", {})),
            }
            for idx, item in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _summarize_evidence(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {"kg": False, "relation": False, "fallback": False}
        for item in candidates:
            evidence = item.get("evidence", {}) or {}
            summary["kg"] = bool(summary["kg"]) or bool(evidence.get("kg"))
            summary["relation"] = bool(summary["relation"]) or bool(evidence.get("relation"))
            summary["fallback"] = bool(summary["fallback"]) or bool(evidence.get("fallback"))
        return summary

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        query = self._normalize_text(text)
        ops = [item["name"] for item in self._rank_entities(query, self.operations, top_k=4, prefix="O")]
        elems = [
            item["name"]
            for item in self._rank_structured_entities(
                query,
                self.elements,
                top_k=4,
                prefix="E",
            )
        ]
        return ops, elems

    def extract_bug_symptoms(self, text: str) -> List[str]:
        query = self._normalize_text(text)
        return [item["name"] for item in self._rank_entities(query, self.bug_symptoms, top_k=4, prefix="B", tail_bias=True)]

    def retrieve_link_candidates(
        self,
        query_text: str,
        top_k_op: int = 4,
        top_k_history: int = 0,
        top_k_instance: int = 4,
    ) -> Dict[str, List[Dict[str, Any]]]:
        query = self._normalize_text(query_text)
        operation_candidates = self._rank_entities(query, self.operations, top_k=top_k_op, prefix="O")
        merged_pages = self._rank_structured_entities(
            query,
            self.pages,
            top_k=top_k_instance,
            prefix="P",
            operation_candidates=operation_candidates,
            relation_map=self.op_to_pages,
        )
        merged_elements = self._rank_structured_entities(
            query,
            self.elements,
            top_k=top_k_instance,
            prefix="E",
            operation_candidates=operation_candidates,
            relation_map=self.op_to_elements,
        )
        direct_bugs = self._rank_entities(
            query,
            self.bug_symptoms,
            top_k=top_k_instance,
            prefix="B",
            tail_bias=True,
        )
        related_bugs = self._rank_related_entities(
            query,
            operation_candidates,
            self.op_to_bugs,
            "B",
            top_k_instance,
            tail_bias=True,
        )
        merged_bugs = self._merge_ranked_entities(direct_bugs, related_bugs, top_k_instance, "B")

        return {
            "operations": operation_candidates,
            "pages": merged_pages,
            "elements": merged_elements,
            "bugs": merged_bugs,
            "historical_reports": [],
            "operation_confidence": self._max_candidate_score(operation_candidates),
            "page_confidence": self._max_candidate_score(merged_pages),
            "element_confidence": self._max_candidate_score(merged_elements),
            "bug_confidence": self._max_candidate_score(merged_bugs),
            "operation_evidence": self._summarize_evidence(operation_candidates),
            "page_evidence": self._summarize_evidence(merged_pages),
            "element_evidence": self._summarize_evidence(merged_elements),
            "bug_evidence": self._summarize_evidence(merged_bugs),
        }

    def retrieve_context(self, query: str, k: int = 3) -> str:
        candidates = self.retrieve_link_candidates(query, top_k_op=k, top_k_instance=k)
        parts: List[str] = []
        if candidates["operations"]:
            parts.append("Operation: " + ", ".join(item["name"] for item in candidates["operations"]))
        if candidates["pages"]:
            parts.append("Page: " + ", ".join(item["name"] for item in candidates["pages"]))
        if candidates["elements"]:
            parts.append("Element: " + ", ".join(item["name"] for item in candidates["elements"]))
        if candidates["bugs"]:
            parts.append("BugSymptom: " + ", ".join(item["name"] for item in candidates["bugs"]))
        return " | ".join(parts)

    def query_severity(self, query: str) -> int:
        normalized = self._normalize_text(query)
        high_keywords = ("崩溃", "闪退", "无响应", "停止运行", "白屏", "黑屏")
        page_keywords = ("页面", "界面", "页")
        if any(keyword in normalized for keyword in high_keywords):
            return 3
        if any(keyword in normalized for keyword in page_keywords):
            return 2
        return 1
