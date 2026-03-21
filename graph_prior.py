"""
GraphPrior pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

from embedding import get_embeddings
from kg_service import get_kg_service
from llm import ask_gpt


PAGE_FAILURE_KEYWORDS = [
    "白屏", "闪退", "卡死", "无法操作", "页面不可用", "整页", "全页面", "崩溃", "crash", "freeze",
]

PROMPT_TEXT_LIMIT = 500
SINGLE_EXTRACTION_MAX_TOKENS = 256
BATCH_EXTRACTION_MAX_TOKENS = 8192
REPRESENTATIVE_SIM_THRESHOLD = 0.90
MISSING_RETRY_BATCH_SIZE = 2
BATCH_RETRY_COVERAGE_THRESHOLD = 0.70
KG_EVIDENCE_TOP_K = 20
HIGH_RISK_SEVERITIES = {"高", "致命"}
PRIORITY_ALPHA = 1.0
PRIORITY_BETA = 1.0
PRIORITY_LAMBDA = 0.5
INSTANCE_MATCH_THRESHOLD = 0.92
OPERATION_MATCH_THRESHOLD = 0.98
TOP_K_OPERATION_CANDIDATES = 2
TOP_K_HISTORY_CANDIDATES = 5
TOP_K_INSTANCE_CANDIDATES = 3
OPERATION_KG_CONFIDENCE_THRESHOLD = 1.2
SLOT_KG_CONFIDENCE_THRESHOLD = 1.6


LINK_SELECTION_PROMPT = """
/no_think
你是缺陷报告链接器。请根据候选列表，为每条报告选择最合适的候选编号。
要求：
1. 只输出纯文本结果，不要解释，不要 Markdown，不要 JSON。
2. OP 只能从 Operation 候选中选 1 个编号；若实在无法判断，可填 NONE。
3. PG / EL 可填候选编号或 NONE。
4. BUG 只能从 Bug 候选中选 1 个编号；若候选都不合适，选最接近的一项。
5. SCOPE 仅允许 element/page/program。
6. 每个报告严格输出一行，格式固定为：
R_001|O1|P0|E2|B3|page
其中 P0 / E0 表示 NONE。
7. 不要输出额外行，不要重复输入内容。

示例输入：
[Report R_000]
TEXT: 点击登录按钮后应用闪退并返回桌面
Operation Candidates:
O1. 点击
O2. 打开
Page Candidates:
P1. 登录页
P2. NONE
Element Candidates:
E1. 登录按钮
E2. 登录入口
Bug Candidates:
B1. 应用闪退
B2. 返回桌面

示例输出：
R_000|O1|P0|E1|B1|program

待处理报告：
{reports_block}
""".strip()


LINK_SELECTION_RETRY_PROMPT = """
/no_think
只按一行一个报告输出，不要解释，不要 Markdown。
格式固定：
R_001|O1|P0|E2|B3|page
其中 P0 / E0 表示 NONE；scope 只能是 element/page/program。

待处理报告：
{reports_block}
""".strip()


class GraphPriorPipeline:
    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        batch_size: int = 16,
        workers: int = 4,
        kg_mode: str = "offline",
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.workers = max(1, int(workers))
        self.kg_mode = kg_mode
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.api_key = api_key
        self.kg_service = get_kg_service(
            model_name=model,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
            kg_mode=kg_mode,
        )
        self.graph = nx.MultiDiGraph()
        self.clusters: List[Dict[str, Any]] = []
        self._instance_store: Dict[str, List[Dict[str, Any]]] = {
            "Operation": [],
            "Page": [],
            "Element": [],
            "Bug": [],
        }
        self._instance_counter = 0

        self.severity_scopes = ["element", "page", "program"]

        # Token usage tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        self.raw_responses: List[Dict[str, Any]] = []
        self._embedding_cache: Dict[str, List[float]] = {}

    def _record_raw_response(
        self,
        call_type: str,
        report_ids: List[str],
        content: str,
        prompt_tokens: int,
        output_tokens: int,
        batch_index: int,
    ) -> None:
        with self.token_lock:
            self.raw_responses.append(
                {
                    "call_type": call_type,
                    "report_ids": report_ids,
                    "content": content,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "batch_index": batch_index,
                }
            )

    def _build_request_options(self) -> Dict[str, Any]:
        model_lower = (self.model or "").lower()
        if "qwen" in model_lower:
            return {
                "extra_body": {
                    "enable_thinking": False,
                    "thinking_budget": 64,
                }
            }
        return {}

    @staticmethod
    def _parse_report_num(report_id: str) -> int:
        match = re.search(r"(\d+)", report_id)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _try_json_load(raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if not raw:
            return {}

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}

        chunk = match.group(0)
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _try_json_list_load(raw: str) -> List[Dict[str, Any]]:
        raw = raw.strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                items = parsed.get("items")
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
                return [parsed]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []

        chunk = match.group(0)
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            return []

        return []

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    @staticmethod
    def _normalize_prompt_text(text: str, limit: int = PROMPT_TEXT_LIMIT) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    @staticmethod
    def _normalize_embedding_text(text: str) -> str:
        normalized = " ".join((text or "").split()).strip()
        return normalized or "未知"

    @staticmethod
    def _resolve_scope(raw_scope: str) -> str:
        scope = (raw_scope or "").strip().lower()
        if "program" in scope:
            return "program"
        if "page" in scope:
            return "page"
        return "element"

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip(" ，。；;,.!?！？:：\"'[]【】()（）")
        normalized = normalized.strip()
        return normalized or "未知"

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            normalized = GraphPriorPipeline._normalize_phrase(item)
            if normalized == "未知" or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _candidate_confidence(candidates: List[Dict[str, Any]]) -> float:
        if not candidates:
            return 0.0
        return max(float(item.get("score", 0.0)) for item in candidates)

    @staticmethod
    def _contains_failure_keyword(text: str) -> bool:
        return any(keyword in text for keyword in PAGE_FAILURE_KEYWORDS)

    def _extract_operation_mentions(self, text: str) -> List[str]:
        if self.kg_service and getattr(self.kg_service, "extract_entities", None):
            ops, _ = self.kg_service.extract_entities(text)
            return self._dedupe_keep_order(ops)
        return []

    def _extract_page_instances(self, text: str) -> List[str]:
        normalized = self._normalize_prompt_text(text, limit=800)
        patterns = [
            r"([^\s，。；,;]{1,16}(?:页面|界面|页面中|界面中|页|目录|栏目|中心))",
            r"(帮助|设置|收藏|所有|最近的|首页|加载页|登录页|关于页|个人中心)(?:页面|界面|页)?",
        ]
        hits: List[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, normalized):
                hits.append(self._normalize_phrase(match))
        return self._dedupe_keep_order(hits)

    def _extract_element_instances(self, text: str) -> List[str]:
        normalized = self._normalize_prompt_text(text, limit=800)
        patterns = [
            r"([^\s，。；,;]{1,16}(?:按钮|图标|星标|星星|入口|菜单栏|菜单|下拉框|输入框|弹窗|对话框|列表|标签|卡片))",
            r"(?:点击|打开|进入|选择|切换|查看|点亮)([^\s，。；,;]{1,12})",
        ]
        hits: List[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, normalized):
                candidate = self._normalize_phrase(match)
                if candidate in {"点击", "打开", "进入", "选择", "切换", "查看", "点亮"}:
                    continue
                hits.append(candidate)
        return self._dedupe_keep_order(hits)

    def _extract_bug_instances(self, text: str) -> List[str]:
        normalized = self._normalize_prompt_text(text, limit=800)
        bug_patterns = [
            r"(无法[^，。；,;]{0,18})",
            r"(不能[^，。；,;]{0,18})",
            r"(不(?:会|能|显示|刷新|更新|生效|响应|支持)[^，。；,;]{0,18})",
            r"(一直在?加载[^，。；,;]{0,18})",
            r"(页面(?:空白|白屏|黑屏)[^，。；,;]{0,12})",
            r"((?:应用|程序|GnuCash)?(?:闪退|崩溃|停止运行|无响应)[^，。；,;]{0,12})",
            r"(报错[^，。；,;]{0,18})",
        ]
        hits: List[str] = []
        for pattern in bug_patterns:
            for match in re.findall(pattern, normalized):
                hits.append(self._normalize_phrase(match))
        if not hits:
            parts = [self._normalize_phrase(part) for part in re.split(r"[。！？!?]", normalized) if part.strip()]
            if parts:
                hits.append(parts[-1])
        return self._dedupe_keep_order(hits)

    def _build_instance_candidates(
        self,
        report_text: str,
        history_candidates: List[Dict[str, Any]],
        memory_label: str,
        extractor,
        top_k: int = TOP_K_INSTANCE_CANDIDATES,
    ) -> List[Dict[str, Any]]:
        support: Dict[str, Dict[str, Any]] = {}

        def add_candidate(name: str, score: float, source: str) -> None:
            normalized = self._normalize_phrase(name)
            if normalized == "未知":
                return
            item = support.setdefault(normalized, {"name": normalized, "score": 0.0, "source": set()})
            item["score"] += score
            item["source"].add(source)

        for item in extractor(report_text):
            add_candidate(item, 2.0, "report")

        for history in history_candidates:
            history_text = str(history.get("description", ""))
            for item in extractor(history_text):
                add_candidate(item, 0.8 + 0.2 * float(history.get("score", 0.0)), "history")

        seeds = extractor(report_text)
        for memory_item in self._instance_store.get(memory_label, []):
            memory_name = str(memory_item.get("name", ""))
            if not memory_name or not seeds:
                continue
            sim = 0.0
            for seed in seeds:
                sim = max(
                    sim,
                    self._cosine_similarity(
                        self._get_cached_embeddings([memory_name])[0],
                        self._get_cached_embeddings([seed])[0],
                    ),
                )
            if sim >= 0.78:
                add_candidate(memory_name, 0.5 + sim, "memory")

        sorted_items = sorted(
            support.values(),
            key=lambda item: (float(item["score"]), len(item["name"])),
            reverse=True,
        )[:top_k]
        return [
            {
                "id": f"{memory_label[0]}{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": ",".join(sorted(item["source"])),
            }
            for idx, item in enumerate(sorted_items, start=1)
        ]

    @staticmethod
    def _merge_candidate_lists(
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
        prefix: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        support: Dict[str, Dict[str, Any]] = {}
        for bucket, boost in ((primary, 1.0), (secondary, 0.7)):
            for item in bucket:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                merged = support.setdefault(name, {"name": name, "score": 0.0, "source": set()})
                merged["score"] += float(item.get("score", 0.0)) + boost
                source = item.get("source", "")
                if source:
                    merged["source"].add(str(source))
        sorted_items = sorted(
            support.values(),
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
            for idx, item in enumerate(sorted_items, start=1)
        ]

    def _build_link_candidates(self, report_id: str, description: str) -> Dict[str, Any]:
        normalized_description = self._normalize_prompt_text(description)
        kg_candidates = {
            "operations": [],
            "pages": [],
            "elements": [],
            "bugs": [],
            "historical_reports": [],
            "operation_confidence": 0.0,
            "page_confidence": 0.0,
            "element_confidence": 0.0,
            "bug_confidence": 0.0,
        }
        if self.kg_service and hasattr(self.kg_service, "retrieve_link_candidates"):
            kg_candidates = self.kg_service.retrieve_link_candidates(
                normalized_description,
                top_k_op=TOP_K_OPERATION_CANDIDATES,
                top_k_history=0,
                top_k_instance=TOP_K_INSTANCE_CANDIDATES,
            )

        operation_candidates = list(kg_candidates.get("operations", []))
        operation_confidence = float(kg_candidates.get("operation_confidence", self._candidate_confidence(operation_candidates)))

        if operation_confidence < OPERATION_KG_CONFIDENCE_THRESHOLD:
            fallback_ops = self._extract_operation_mentions(normalized_description) or ["点击"]
            operation_candidates = [
                {"id": f"O{idx}", "name": name, "score": float(len(fallback_ops) - idx + 1), "source": "fallback"}
                for idx, name in enumerate(fallback_ops[:TOP_K_OPERATION_CANDIDATES], start=1)
            ]
            operation_confidence = self._candidate_confidence(operation_candidates)

        local_page_candidates = self._build_instance_candidates(
            normalized_description,
            [],
            memory_label="Page",
            extractor=self._extract_page_instances,
        )
        local_element_candidates = self._build_instance_candidates(
            normalized_description,
            [],
            memory_label="Element",
            extractor=self._extract_element_instances,
        )
        local_bug_candidates = self._build_instance_candidates(
            normalized_description,
            [],
            memory_label="Bug",
            extractor=self._extract_bug_instances,
        )

        kg_page_candidates = list(kg_candidates.get("pages", [])) if float(kg_candidates.get("page_confidence", 0.0)) >= SLOT_KG_CONFIDENCE_THRESHOLD else []
        kg_element_candidates = list(kg_candidates.get("elements", [])) if float(kg_candidates.get("element_confidence", 0.0)) >= SLOT_KG_CONFIDENCE_THRESHOLD else []
        kg_bug_candidates = list(kg_candidates.get("bugs", [])) if float(kg_candidates.get("bug_confidence", 0.0)) >= SLOT_KG_CONFIDENCE_THRESHOLD else []

        page_candidates = self._merge_candidate_lists(
            kg_page_candidates,
            local_page_candidates,
            prefix="P",
            top_k=TOP_K_INSTANCE_CANDIDATES,
        )
        element_candidates = self._merge_candidate_lists(
            kg_element_candidates,
            local_element_candidates,
            prefix="E",
            top_k=TOP_K_INSTANCE_CANDIDATES,
        )
        bug_candidates = self._merge_candidate_lists(
            kg_bug_candidates,
            local_bug_candidates,
            prefix="B",
            top_k=TOP_K_INSTANCE_CANDIDATES,
        )

        if not page_candidates:
            page_candidates = [{"id": "P1", "name": "未知页面", "score": 0.1, "source": "fallback"}]
        if not element_candidates:
            element_candidates = [{"id": "E1", "name": "未知组件", "score": 0.1, "source": "fallback"}]
        if not bug_candidates:
            bug_candidates = [{"id": "B1", "name": self._normalize_phrase(description), "score": 0.1, "source": "fallback"}]

        return {
            "report_id": report_id,
            "text": normalized_description,
            "operations": operation_candidates,
            "pages": page_candidates,
            "elements": element_candidates,
            "bugs": bug_candidates,
            "operation_confidence": operation_confidence,
            "page_confidence": self._candidate_confidence(kg_page_candidates),
            "element_confidence": self._candidate_confidence(kg_element_candidates),
            "bug_confidence": self._candidate_confidence(kg_bug_candidates),
        }

    def _format_candidates(self, title: str, candidates: List[Dict[str, Any]]) -> str:
        if not candidates:
            return f"{title}:\nNONE"
        lines = [f"{title}:"]
        for item in candidates:
            suffix = ""
            if "severity" in item and item.get("severity"):
                suffix = f" | severity={item['severity']}"
            lines.append(f"{item['id']}. {item['name'] if 'name' in item else item['description'][:80]}{suffix}")
        return "\n".join(lines)

    def _build_candidate_report_block(self, candidate_payloads: List[Dict[str, Any]]) -> str:
        blocks = []
        for payload in candidate_payloads:
            blocks.append(
                "\n".join(
                    [
                        f"[Report {payload['report_id']}]",
                        f"TEXT: {payload['text']}",
                        self._format_candidates("Operation Candidates", payload["operations"]),
                        self._format_candidates("Page Candidates", payload["pages"]),
                        self._format_candidates("Element Candidates", payload["elements"]),
                        self._format_candidates("Bug Candidates", payload["bugs"]),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _batch_max_tokens(self, report_count: int, retry: bool = False) -> int:
        budget = min(BATCH_EXTRACTION_MAX_TOKENS, SINGLE_EXTRACTION_MAX_TOKENS * max(1, report_count))
        return budget

    @staticmethod
    def _get_field(data: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = data.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _select_candidate_name(
        self,
        raw_value: str,
        candidates: List[Dict[str, Any]],
        fallback: str,
    ) -> str:
        value = self._normalize_phrase(raw_value)
        if not value or value.upper() in {"NONE", "P0", "E0", "O0", "B0"}:
            return fallback
        for item in candidates:
            if value == str(item.get("id", "")).strip():
                return self._normalize_phrase(str(item.get("name", fallback)))
        return fallback

    def _normalize_link_selection_item(
        self,
        data: Dict[str, Any],
        candidate_payload: Dict[str, Any],
        description: str,
    ) -> Dict[str, str]:
        report_id = candidate_payload["report_id"]
        operation = self._select_candidate_name(
            self._get_field(data, "Operation", "op"),
            candidate_payload["operations"],
            fallback=self._normalize_phrase(candidate_payload["operations"][0]["name"]) if candidate_payload["operations"] else "未知操作",
        )
        page = self._select_candidate_name(
            self._get_field(data, "Page", "pg"),
            candidate_payload["pages"],
            fallback="未知页面",
        )
        element = self._select_candidate_name(
            self._get_field(data, "Element", "el"),
            candidate_payload["elements"],
            fallback="未知组件",
        )
        bug = self._select_candidate_name(
            self._get_field(data, "Bug", "bug"),
            candidate_payload["bugs"],
            fallback=self._normalize_phrase(description),
        )
        severity_scope = self._resolve_scope(self._get_field(data, "Severity_Scope", "scope"))

        return {
            "Report_ID": report_id,
            "Operation": operation,
            "Page": page,
            "Element": element,
            "Bug": bug,
            "Severity_Scope": severity_scope,
            "Reference_Report": self._get_field(data, "Reference_Report", "ref"),
        }

    @staticmethod
    def _try_block_load(raw: str) -> List[Dict[str, Any]]:
        raw = (raw or "").strip()
        if not raw:
            return []

        items: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        field_map = {
            "ID": "id",
            "OP": "op",
            "PG": "pg",
            "EL": "el",
            "BUG": "bug",
            "REF": "ref",
            "SCOPE": "scope",
        }

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == "###":
                if current.get("id"):
                    items.append(current)
                current = {}
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            mapped = field_map.get(key.strip().upper())
            if mapped:
                if mapped == "id" and current.get("id"):
                    items.append(current)
                    current = {}
                current[mapped] = value.strip()

        if current.get("id"):
            items.append(current)
        return items

    @staticmethod
    def _try_compact_line_load(raw: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for line in (raw or "").splitlines():
            compact = line.strip()
            if not compact or "|" not in compact or not compact.startswith("R_"):
                continue
            parts = [part.strip() for part in compact.split("|")]
            if len(parts) != 6:
                continue
            report_id, op, pg, el, bug, scope = parts
            items.append(
                {
                    "id": report_id,
                    "op": op,
                    "pg": "NONE" if pg.upper() == "P0" else pg,
                    "el": "NONE" if el.upper() == "E0" else el,
                    "bug": "NONE" if bug.upper() == "B0" else bug,
                    "scope": scope,
                }
            )
        return items

    def _parse_extraction_items(self, raw: str) -> List[Dict[str, Any]]:
        compact_items = self._try_compact_line_load(raw)
        if compact_items:
            return compact_items

        block_items = self._try_block_load(raw)
        if block_items:
            return block_items

        json_list = self._try_json_list_load(raw)
        if json_list:
            return json_list

        json_item = self._try_json_load(raw)
        if json_item:
            return [json_item]
        return []

    @staticmethod
    def _chunk_pairs(items: List[Any], chunk_size: int) -> List[List[Any]]:
        if not items:
            return []
        return [items[start:start + chunk_size] for start in range(0, len(items), chunk_size)]

    def _get_cached_embeddings(self, texts: List[str]) -> List[List[float]]:
        normalized_texts = [self._normalize_embedding_text(text) for text in texts]
        ordered_missing: List[str] = []
        seen_missing = set()

        with self.embedding_lock:
            for text in normalized_texts:
                if text not in self._embedding_cache and text not in seen_missing:
                    ordered_missing.append(text)
                    seen_missing.add(text)

            if ordered_missing:
                vectors = get_embeddings(
                    ordered_missing,
                    model=self.model,
                    embedding_model=self.embedding_model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
                if len(vectors) != len(ordered_missing):
                    raise RuntimeError(
                        f"Embedding generation failed. expected={len(ordered_missing)}, got={len(vectors)}"
                    )
                for idx, text in enumerate(ordered_missing):
                    self._embedding_cache[text] = vectors[idx].tolist()

            return [list(self._embedding_cache[text]) for text in normalized_texts]

    def _group_reports_for_extraction(
        self,
        indexed_reports: List[Tuple[int, Dict[str, Any]]],
    ) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
        if not indexed_reports:
            return {}

        description_vectors = get_embeddings(
            [self._normalize_prompt_text(report.get("description", "")) for _, report in indexed_reports],
            model=self.model,
            embedding_model=self.embedding_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        groups: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        representative_positions: List[int] = []

        for pos, (report_num, report) in enumerate(indexed_reports):
            current_vec = description_vectors[pos].tolist()
            matched_rep_num: Optional[int] = None
            best_sim = 0.0

            for rep_pos in representative_positions:
                rep_vec = description_vectors[rep_pos].tolist()
                sim = self._cosine_similarity(current_vec, rep_vec)
                if sim >= REPRESENTATIVE_SIM_THRESHOLD and sim > best_sim:
                    matched_rep_num = indexed_reports[rep_pos][0]
                    best_sim = sim

            if matched_rep_num is None:
                representative_positions.append(pos)
                groups[report_num] = [(report_num, report)]
            else:
                groups[matched_rep_num].append((report_num, report))

        return groups

    def _build_priority_query_text(self, rep_row: Dict[str, Any]) -> str:
        parts = []
        for key in ("Operation", "Page", "Element", "Bug"):
            value = self._normalize_prompt_text(str(rep_row.get(key, "")))
            if value and not value.startswith("未知"):
                parts.append(value)
        return " ".join(parts)

    def _get_representation_text(self, rep_row: Dict[str, Any]) -> str:
        parts = [
            self._normalize_prompt_text(str(rep_row.get("Bug", ""))),
            self._normalize_prompt_text(str(rep_row.get("Page", ""))),
            self._normalize_prompt_text(str(rep_row.get("Element", ""))),
        ]
        return " ".join(part for part in parts if part and not part.startswith("未知"))

    def _get_available_priority_slots(self, rep_row: Dict[str, Any]) -> int:
        available = 0
        if str(rep_row.get("Operation", "")).strip() and not str(rep_row.get("Operation", "")).startswith("未知"):
            available += 1
        if str(rep_row.get("Element", "")).strip() and not str(rep_row.get("Element", "")).startswith("未知"):
            available += 1
        if str(rep_row.get("Page", "")).strip() and not str(rep_row.get("Page", "")).startswith("未知"):
            available += 1
        if str(rep_row.get("Bug", "")).strip() and not str(rep_row.get("Bug", "")).startswith("未知"):
            available += 1
        return max(1, available)

    def _compute_priority_from_evidence(
        self,
        cluster: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> float:
        cluster_size = 1 + len(cluster.get("dups", []))
        rep_row = cluster["rep"]
        if not evidence:
            risk = 0.5
            support = 1.0
        else:
            available_slots = self._get_available_priority_slots(rep_row)
            weighted_total = 0.0
            weighted_high_risk = 0.0

            for item in evidence:
                matched_slots = 0
                if item.get("matched_ops"):
                    matched_slots += 1
                if item.get("matched_elements"):
                    matched_slots += 1
                if item.get("matched_pages"):
                    matched_slots += 1
                if item.get("matched_bugs"):
                    matched_slots += 1

                overlap_ratio = matched_slots / float(available_slots)
                weight = max(0.0, float(item.get("vector_score", 0.0))) * (1.0 + PRIORITY_ALPHA * overlap_ratio)
                weighted_total += weight
                if str(item.get("severity", "")).strip() in HIGH_RISK_SEVERITIES:
                    weighted_high_risk += weight

            if weighted_total <= 0:
                risk = 0.5
                support = 1.0
            else:
                risk = weighted_high_risk / weighted_total
                support = math.log1p(weighted_total)

        cluster_support = math.pow(math.log1p(cluster_size), PRIORITY_BETA)
        base_priority = float(risk * support * cluster_support)
        cluster["risk_score"] = risk
        cluster["support_score"] = support
        cluster["cluster_support"] = cluster_support
        cluster["cluster_score"] = base_priority
        return base_priority

    def _kg_priority_scoring(self) -> Optional[Dict[int, float]]:
        if (
            not self.kg_service
            or not hasattr(self.kg_service, "retrieve_priority_evidence")
            or not getattr(self.kg_service, "supports_priority_evidence", True)
        ):
            return None

        score_map: Dict[int, float] = {}
        for cluster in self.clusters:
            rep_row = cluster["rep"]
            evidence = self.kg_service.retrieve_priority_evidence(
                query_text=self._build_priority_query_text(rep_row),
                op_text=str(rep_row.get("Operation", "")),
                page_text=str(rep_row.get("Page", "")),
                elem_text=str(rep_row.get("Element", "")),
                bug_text=str(rep_row.get("Bug", "")),
                top_k=KG_EVIDENCE_TOP_K,
            )
            cluster["priority_evidence"] = evidence
            score = self._compute_priority_from_evidence(cluster, evidence)
            score_map[int(rep_row["report_num"])] = score
            for dup in cluster.get("dups", []):
                score_map[int(dup["report_num"])] = score

            report_node = f"Report:{rep_row['Report_ID']}"
            if self.graph.has_node(report_node):
                self.graph.nodes[report_node]["severity_score"] = score

        return score_map

    def _rank_clusters_with_diversity(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not clusters:
            return []

        rep_texts = [self._get_representation_text(cluster["rep"]) or cluster["rep"].get("Bug", "") for cluster in clusters]
        vectors = self._get_cached_embeddings(rep_texts)
        remaining = list(range(len(clusters)))
        selected: List[int] = []

        while remaining:
            best_idx = remaining[0]
            best_score = -1.0

            for idx in remaining:
                base_priority = float(clusters[idx].get("cluster_score", 0.0))
                similarity_penalty = 0.0
                if selected:
                    similarity_penalty = max(
                        self._cosine_similarity(vectors[idx], vectors[chosen_idx])
                        for chosen_idx in selected
                    )
                priority = base_priority / (1.0 + PRIORITY_LAMBDA * max(0.0, similarity_penalty))
                if priority > best_score:
                    best_idx = idx
                    best_score = priority

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [clusters[idx] for idx in selected]

    def _render_link_prompt(self, candidate_payloads: List[Dict[str, Any]], compact_only: bool = False) -> str:
        prompt_template = LINK_SELECTION_RETRY_PROMPT if compact_only else LINK_SELECTION_PROMPT
        return prompt_template.format(reports_block=self._build_candidate_report_block(candidate_payloads))

    def _extract_structured(
        self,
        candidate_payload: Dict[str, Any],
        batch_index: int = -1,
        compact_only: bool = False,
    ) -> Dict[str, str]:
        report_id = candidate_payload["report_id"]
        prompt = self._render_link_prompt([candidate_payload], compact_only=compact_only)
        content, p_tokens, r_tokens = ask_gpt(
            prompt,
            model=self.model,
            max_token=SINGLE_EXTRACTION_MAX_TOKENS,
            base_url=self.base_url,
            api_key=self.api_key,
            request_options=self._build_request_options(),
        )
        with self.token_lock:
            self.input_tokens += p_tokens
            self.output_tokens += r_tokens
        self._record_raw_response(
            call_type="single_retry" if compact_only else "single",
            report_ids=[report_id],
            content=content,
            prompt_tokens=p_tokens,
            output_tokens=r_tokens,
            batch_index=batch_index,
        )
        parsed_items = self._parse_extraction_items(content)
        parsed = parsed_items[0] if parsed_items else {}
        return self._normalize_link_selection_item(
            parsed,
            candidate_payload=candidate_payload,
            description=str(candidate_payload.get("text", "")),
        )

    def _extract_structured_batch(
        self,
        candidate_payloads: List[Dict[str, Any]],
        batch_index: int,
        compact_only: bool = False,
        call_type: str = "batch",
    ) -> Dict[str, Dict[str, str]]:
        report_ids = [payload["report_id"] for payload in candidate_payloads]
        prompt = self._render_link_prompt(candidate_payloads, compact_only=compact_only)
        content, p_tokens, r_tokens = ask_gpt(
            prompt,
            model=self.model,
            max_token=self._batch_max_tokens(len(candidate_payloads)),
            base_url=self.base_url,
            api_key=self.api_key,
            request_options=self._build_request_options(),
        )
        with self.token_lock:
            self.input_tokens += p_tokens
            self.output_tokens += r_tokens
        self._record_raw_response(
            call_type=call_type,
            report_ids=report_ids,
            content=content,
            prompt_tokens=p_tokens,
            output_tokens=r_tokens,
            batch_index=batch_index,
        )

        parsed_by_id: Dict[str, Dict[str, Any]] = {}
        for item in self._parse_extraction_items(content):
            rid = self._get_field(item, "Report_ID", "id")
            if rid and rid not in parsed_by_id:
                parsed_by_id[rid] = item

        normalized: Dict[str, Dict[str, str]] = {}
        for payload in candidate_payloads:
            report_id = payload["report_id"]
            parsed = parsed_by_id.get(report_id)
            if not parsed:
                continue
            normalized[report_id] = self._normalize_link_selection_item(
                parsed,
                candidate_payload=payload,
                description=str(payload.get("text", "")),
            )
        return normalized

    @staticmethod
    def _coverage_ratio(candidate_payloads: List[Dict[str, Any]], extracted_batch: Dict[str, Dict[str, str]]) -> float:
        total = max(1, len(candidate_payloads))
        covered = sum(1 for payload in candidate_payloads if payload["report_id"] in extracted_batch)
        return covered / float(total)

    def _fill_missing_extractions(
        self,
        candidate_payloads: List[Dict[str, Any]],
        extracted_batch: Dict[str, Dict[str, str]],
        batch_index: int,
    ) -> Dict[str, Dict[str, str]]:
        coverage = self._coverage_ratio(candidate_payloads, extracted_batch)
        if len(candidate_payloads) > 1 and coverage < BATCH_RETRY_COVERAGE_THRESHOLD:
            retry_batch = self._extract_structured_batch(
                candidate_payloads,
                batch_index=batch_index,
                compact_only=True,
                call_type="batch_retry",
            )
            for report_id, item in retry_batch.items():
                extracted_batch[report_id] = item

        missing_reports = [payload for payload in candidate_payloads if payload["report_id"] not in extracted_batch]
        if not missing_reports:
            return extracted_batch

        for retry_batch in self._chunk_pairs(
            [item for item in missing_reports if item["report_id"] not in extracted_batch],
            MISSING_RETRY_BATCH_SIZE,
        ):
            if len(retry_batch) == 1:
                break
            extracted_batch.update(
                self._extract_structured_batch(
                    retry_batch,
                    batch_index=batch_index,
                    compact_only=True,
                    call_type="batch_missing",
                )
            )

        for payload in candidate_payloads:
            report_id = payload["report_id"]
            if report_id in extracted_batch:
                continue
            extracted_batch[report_id] = self._extract_structured(
                payload,
                batch_index=batch_index,
                compact_only=True,
            )
        return extracted_batch

    def _next_instance_id(self, label: str) -> str:
        self._instance_counter += 1
        return f"{label}:{self._instance_counter}"

    def _match_or_create_instance_dimension(self, label: str, name: str, embedding: List[float], threshold: float = 0.9) -> str:
        candidates = self._instance_store.setdefault(label, [])

        for item in candidates:
            if item["name"] == name:
                return item["id"]

        for item in candidates:
            sim = self._cosine_similarity(item["embedding"], embedding)
            if sim > threshold:
                return item["id"]

        new_id = self._next_instance_id(label)
        candidates.append({"id": new_id, "name": name, "embedding": embedding})
        self.graph.add_node(new_id, label=label, name=name, embedding=embedding)
        return new_id

    def _prepare_chunk(
        self,
        batch_index: int,
        chunk: List[Tuple[int, Dict[str, Any]]],
        grouped_reports: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        candidate_payloads: List[Dict[str, Any]] = []
        for idx, report in chunk:
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            candidate_payloads.append(self._build_link_candidates(report_id, description))

        extracted_batch = self._extract_structured_batch(candidate_payloads, batch_index=batch_index)
        extracted_batch = self._fill_missing_extractions(candidate_payloads, extracted_batch, batch_index=batch_index)

        records: List[Tuple[int, str, str, Dict[str, str]]] = []
        embedding_texts: List[str] = []
        for rep_idx, _report in chunk:
            rep_report_id = f"R_{rep_idx:03d}"
            ext = extracted_batch.get(rep_report_id)
            if not ext:
                payload = next((item for item in candidate_payloads if item["report_id"] == rep_report_id), None)
                if payload is None:
                    continue
                ext = self._normalize_link_selection_item({}, candidate_payload=payload, description=str(payload.get("text", "")))
            for idx, report in grouped_reports.get(rep_idx, []):
                report_id = f"R_{idx:03d}"
                description = report.get("description", "")
                records.append((idx, report_id, description, ext))
                embedding_texts.extend([
                    ext["Operation"],
                    ext["Page"],
                    ext["Element"],
                    ext["Bug"],
                ])

        vectors = self._get_cached_embeddings(embedding_texts)
        expected = len(records) * 4
        if len(vectors) != expected:
            raise RuntimeError(f"Embedding generation failed for chunk. expected={expected}, got={len(vectors)}")

        prepared: List[Dict[str, Any]] = []
        for i, (idx, report_id, description, ext) in enumerate(records):
            base = i * 4
            prepared.append(
                {
                    "report_num": idx,
                    "report_id": report_id,
                    "description": description,
                    "ext": ext,
                    "v_op": vectors[base],
                    "v_pg": vectors[base + 1],
                    "v_el": vectors[base + 2],
                    "v_bg": vectors[base + 3],
                }
            )
        return prepared

    def phase1_extract_and_fuse(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        indexed_reports = list(enumerate(reports, start=1))
        grouped_reports = self._group_reports_for_extraction(indexed_reports)
        representative_reports = [
            grouped_reports[rep_num][0]
            for rep_num in sorted(grouped_reports.keys())
        ]
        batch_size = self.batch_size

        chunks: List[List[Tuple[int, Dict[str, Any]]]] = [
            representative_reports[start:start + batch_size]
            for start in range(0, len(representative_reports), batch_size)
        ]

        prepared_all: List[Dict[str, Any]] = []
        if self.workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [
                    executor.submit(self._prepare_chunk, batch_index, chunk, grouped_reports)
                    for batch_index, chunk in enumerate(chunks, start=1)
                ]
                for future in as_completed(futures):
                    prepared_all.extend(future.result())
        else:
            for batch_index, chunk in enumerate(chunks, start=1):
                prepared_all.extend(self._prepare_chunk(batch_index, chunk, grouped_reports))

        extracted_rows: List[Dict[str, Any]] = []
        prepared_all.sort(key=lambda x: int(x["report_num"]))

        for item in prepared_all:
            idx = int(item["report_num"])
            report_id = str(item["report_id"])
            description = str(item["description"])
            ext = item["ext"]

            op_text = ext["Operation"]
            page_text = ext["Page"]
            elem_text = ext["Element"]
            bug_text = ext["Bug"]
            severity_scope = ext["Severity_Scope"]

            v_op = item["v_op"]
            v_pg = item["v_pg"]
            v_el = item["v_el"]
            v_bg = item["v_bg"]

            op_eid = self._match_or_create_instance_dimension("Operation", op_text, v_op, threshold=OPERATION_MATCH_THRESHOLD)
            page_eid = self._match_or_create_instance_dimension("Page", page_text, v_pg, threshold=INSTANCE_MATCH_THRESHOLD)
            elem_eid = self._match_or_create_instance_dimension("Element", elem_text, v_el, threshold=INSTANCE_MATCH_THRESHOLD)
            bug_eid = self._match_or_create_instance_dimension("Bug", bug_text, v_bg, threshold=INSTANCE_MATCH_THRESHOLD)

            signature = (op_eid, page_eid, elem_eid, bug_eid)
            
            report_node = f"Report:{report_id}"
            self.graph.add_node(
                report_node,
                label="Report",
                id=report_id,
                report_num=idx,
                description=description,
                severity_scope=severity_scope,
            )
            
            self.graph.add_edge(report_node, op_eid, rel="TRIGGERED_BY")
            self.graph.add_edge(report_node, page_eid, rel="OCCURS_ON_PAGE")
            self.graph.add_edge(report_node, elem_eid, rel="OCCURS_ON_ELEMENT")
            self.graph.add_edge(report_node, bug_eid, rel="MANIFESTS_AS")
            self.graph.add_edge(elem_eid, page_eid, rel="LOCATED_ON")

            row_data = {
                **ext,
                "report_num": idx,
                "operation_eid": op_eid,
                "page_eid": page_eid,
                "element_eid": elem_eid,
                "bug_eid": bug_eid,
                "signature": signature,
                "description": description,
            }
            extracted_rows.append(row_data)

            cluster_found = False
            for cluster in self.clusters:
                if cluster["signature"] == signature:
                    cluster["dups"].append(row_data)
                    cluster_found = True
                    break
            
            if not cluster_found:
                self.clusters.append({
                    "signature": signature,
                    "rep": row_data,
                    "dups": [],
                    "severity_scope": severity_scope,
                })

        return extracted_rows

    def phase2_topology_scoring(self) -> Dict[int, float]:
        kg_score_map = self._kg_priority_scoring()
        if kg_score_map is not None:
            return kg_score_map

        score_map: Dict[int, float] = {}

        # 1. 统计所有的组件数 I (Program 级影响) 以及具体页面的组件映射
        # 注意: _instance_store["Element"] 中保存了本次处理的所有元素，这里我们利用图边获取更准确的从属关系
        total_elements = len(self._instance_store.get("Element", []))
        
        page_to_elements = {}
        for u, v, data in self.graph.edges(data=True):
            if data.get("rel") == "LOCATED_ON":
                # u: element_eid, v: page_eid
                page_to_elements.setdefault(v, set()).add(u)

        # 2. 依次计算每个簇的分数
        for cluster in self.clusters:
            rep_row = cluster["rep"]
            rep_num = int(rep_row["report_num"])
            severity_scope = cluster.get("severity_scope", "element")
            t = 1 + len(cluster.get("dups", []))  # T: 报告出现次数

            # 确定影响组件数 I
            i_val = 1
            if severity_scope == "program":
                i_val = max(1, total_elements)
            elif severity_scope == "page":
                page_eid = rep_row.get("page_eid")
                i_val = max(1, len(page_to_elements.get(page_eid, set())))
            
            # 计算得分 score = I * log(T + 1)
            final_score = float(i_val * math.log(t + 1))
            cluster["cluster_score"] = final_score
            
            # 记录簇代表与其所有重复报告的分数
            score_map[rep_num] = final_score
            for dup in cluster.get("dups", []):
                score_map[int(dup["report_num"])] = final_score

            # 将分数写回图谱报告节点
            report_node = f"Report:{rep_row['Report_ID']}"
            if self.graph.has_node(report_node):
                self.graph.nodes[report_node]["severity_score"] = final_score

        return score_map

    def run(self, reports: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, float], int, int, List[Dict[str, Any]]]:
        self.phase1_extract_and_fuse(reports)
        score_map = self.phase2_topology_scoring()

        # Sort cluster representatives with diversity-aware reranking.
        sorted_clusters = self._rank_clusters_with_diversity(self.clusters)
        
        round1 = []
        round2 = []

        for cluster in sorted_clusters:
            round1.append(int(cluster["rep"]["report_num"]))
            # Duplicates are also sorted locally (though they share the same score here logic-wise)
            for dup in cluster.get("dups", []):
                round2.append(int(dup["report_num"]))

        sequence = round1 + round2
        raw_responses = sorted(
            self.raw_responses,
            key=lambda item: (int(item.get("batch_index", -1)), 0 if item.get("call_type") == "batch" else 1),
        )
        return sequence, score_map, self.input_tokens, self.output_tokens, raw_responses


def run_graph_prior(
    reports: List[Dict[str, Any]],
    model: str,
    batch_size: int = 16,
    workers: int = 4,
    kg_mode: str = "offline",
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[int], Dict[int, float], int, int, List[Dict[str, Any]]]:
    pipeline = GraphPriorPipeline(
        model=model,
        batch_size=batch_size,
        workers=workers,
        kg_mode=kg_mode,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )
    return pipeline.run(reports)
