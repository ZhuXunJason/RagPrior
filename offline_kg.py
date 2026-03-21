"""
Offline KG service backed by local KG_desc.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re


@dataclass
class OfflineKG:
    bug_symptoms: List[str]
    operations: List[str]
    severities: List[str]
    elements: List[str]
    pages: List[str]
    op_to_elements: Dict[str, List[str]]
    op_to_pages: Dict[str, List[str]]
    op_to_bugs: Dict[str, List[str]]


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


def load_offline_kg(desc_path: str | Path) -> OfflineKG:
    path = Path(desc_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: Dict[str, List[str]] = {
        "bug_symptoms": [],
        "operations": [],
        "severities": [],
        "elements": [],
        "pages": [],
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
            sections["elements"], index = _parse_section(lines, index)
            continue
        if line.startswith("### Page"):
            sections["pages"], index = _parse_section(lines, index)
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

    def _score_entity(self, query: str, entity: str) -> float:
        query_norm = self._normalize_text(query)
        entity_norm = self._normalize_text(entity)
        if not query_norm or not entity_norm:
            return 0.0

        score = 0.0
        if entity_norm in query_norm:
            score += 3.0
        if query_norm in entity_norm:
            score += 1.0

        query_base = self._strip_suffixes(query_norm)
        entity_base = self._strip_suffixes(entity_norm)
        if entity_base and entity_base in query_norm:
            score += 1.5
        if query_base and query_base in entity_norm:
            score += 0.8

        score += self._char_overlap_ratio(query_norm, entity_norm)
        return score

    def _rank_entities(self, query: str, entities: List[str], top_k: int, prefix: str) -> List[Dict[str, Any]]:
        scored = []
        for entity in entities:
            score = self._score_entity(query, entity)
            if score < 0.45:
                continue
            scored.append({"name": entity, "score": score})
        scored.sort(key=lambda item: (float(item["score"]), len(item["name"])), reverse=True)
        return [
            {"id": f"{prefix}{idx}", "name": item["name"], "score": float(item["score"]), "source": "offline-kg"}
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
    ) -> List[Dict[str, Any]]:
        related_scores: Dict[str, Dict[str, Any]] = {}
        for op_item in operation_candidates:
            op_name = str(op_item.get("name", "")).strip()
            op_score = float(op_item.get("score", 0.0))
            for related_name in relation_map.get(op_name, []):
                direct_score = self._score_entity(query, related_name)
                total_score = 0.7 * op_score + direct_score
                if total_score < 0.6:
                    continue
                entry = related_scores.setdefault(
                    related_name,
                    {"name": related_name, "score": 0.0, "source": set()},
                )
                entry["score"] += total_score
                entry["source"].add(f"offline-rel:{op_name}")
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
            }
            for idx, item in enumerate(ranked, start=1)
        ]

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        query = self._normalize_text(text)
        ops = [item["name"] for item in self._rank_entities(query, self.operations, top_k=4, prefix="O")]
        elems = [item["name"] for item in self._rank_entities(query, self.elements, top_k=4, prefix="E")]
        return ops, elems

    def extract_bug_symptoms(self, text: str) -> List[str]:
        query = self._normalize_text(text)
        return [item["name"] for item in self._rank_entities(query, self.bug_symptoms, top_k=4, prefix="B")]

    def retrieve_link_candidates(
        self,
        query_text: str,
        top_k_op: int = 4,
        top_k_history: int = 0,
        top_k_instance: int = 4,
    ) -> Dict[str, List[Dict[str, Any]]]:
        query = self._normalize_text(query_text)
        operation_candidates = self._rank_entities(query, self.operations, top_k=top_k_op, prefix="O")
        direct_pages = self._rank_entities(query, self.pages, top_k=top_k_instance, prefix="P")
        direct_elements = self._rank_entities(query, self.elements, top_k=top_k_instance, prefix="E")
        direct_bugs = self._rank_entities(query, self.bug_symptoms, top_k=top_k_instance, prefix="B")

        related_pages = self._rank_related_entities(query, operation_candidates, self.op_to_pages, "P", top_k_instance)
        related_elements = self._rank_related_entities(query, operation_candidates, self.op_to_elements, "E", top_k_instance)
        related_bugs = self._rank_related_entities(query, operation_candidates, self.op_to_bugs, "B", top_k_instance)

        merged_pages = self._merge_ranked_entities(direct_pages, related_pages, top_k_instance, "P")
        merged_elements = self._merge_ranked_entities(direct_elements, related_elements, top_k_instance, "E")
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
