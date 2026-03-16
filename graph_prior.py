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
BATCH_EXTRACTION_MAX_TOKENS = 2048
RETRY_BATCH_EXTRACTION_MAX_TOKENS = 4096


EXTRACTION_PROMPT = """
你是缺陷报告抽取器，只输出单行 JSON，不要解释，不要 Markdown，不要代码块。
字段必须是：
{{"id":"{report_id}","op":"操作","pg":"页面","el":"组件","bug":"缺陷现象","scope":"element|page|program"}}
规则：
1. 信息缺失时分别填：op=未知操作, pg=未知页面, el=未知组件, bug=未知缺陷。
2. pg 和 el 不要同时无意义；若页面无法确定，尽量提取业务组件到 el。
3. scope 仅允许 element/page/program。
输入：
{report_id}\t{text}
""".strip()


EXTRACTION_BATCH_PROMPT = """
你是缺陷报告抽取器，只输出单行 JSON 数组，不要解释，不要 Markdown，不要代码块，不要 pretty print。
每个对象字段必须是：
[{{"id":"R_001","op":"操作","pg":"页面","el":"组件","bug":"缺陷现象","scope":"element|page|program"}}]
规则：
1. 信息缺失时分别填：op=未知操作, pg=未知页面, el=未知组件, bug=未知缺陷。
2. pg 和 el 不要同时无意义；若页面无法确定，尽量提取业务组件到 el。
3. scope 仅允许 element/page/program。
4. 必须覆盖输入中的每条报告，各 id 只能出现一次。

示例输入：
R_000\t点击登录按钮后应用闪退并返回桌面
示例输出：
[{{"id":"R_000","op":"点击","pg":"登录页","el":"登录按钮","bug":"应用闪退","scope":"program"}}]

待处理报告：
{reports_block}
""".strip()


class GraphPriorPipeline:
    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        batch_size: int = 16,
        workers: int = 4,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.workers = max(1, int(workers))
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.api_key = api_key
        self.kg_service = get_kg_service(
            model_name=model,
            embedding_model=embedding_model,
            base_url=base_url,
            api_key=api_key,
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

    def _build_batch_reports_block(self, reports_batch: List[Tuple[str, str]]) -> str:
        return "\n".join(
            f"{report_id}\t{self._normalize_prompt_text(text)}"
            for report_id, text in reports_batch
        )

    def _batch_max_tokens(self, report_count: int, retry: bool = False) -> int:
        budget = min(BATCH_EXTRACTION_MAX_TOKENS, SINGLE_EXTRACTION_MAX_TOKENS * max(1, report_count))
        if retry:
            budget = min(RETRY_BATCH_EXTRACTION_MAX_TOKENS, budget * 2)
        return budget

    @staticmethod
    def _get_field(data: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = data.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_extraction_item(
        self,
        data: Dict[str, Any],
        report_id: str,
        description: str,
    ) -> Dict[str, str]:
        operation = self._get_field(data, "Operation", "op") or "未知操作"
        page = self._get_field(data, "Page", "pg") or "未知页面"
        element = self._get_field(data, "Element", "el") or "未知组件"
        bug = self._get_field(data, "Bug", "bug") or description
        severity_scope = self._resolve_scope(self._get_field(data, "Severity_Scope", "scope"))

        return {
            "Report_ID": report_id,
            "Operation": operation,
            "Page": page,
            "Element": element,
            "Bug": bug,
            "Severity_Scope": severity_scope,
        }

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

    def _extract_structured(self, report_id: str, description: str, batch_index: int = -1) -> Dict[str, str]:
        normalized_description = self._normalize_prompt_text(description)
        prompt = EXTRACTION_PROMPT.format(report_id=report_id, text=normalized_description)
        content, p_tokens, r_tokens = ask_gpt(
            prompt,
            model=self.model,
            max_token=SINGLE_EXTRACTION_MAX_TOKENS,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        with self.token_lock:
            self.input_tokens += p_tokens
            self.output_tokens += r_tokens
        self._record_raw_response(
            call_type="single",
            report_ids=[report_id],
            content=content,
            prompt_tokens=p_tokens,
            output_tokens=r_tokens,
            batch_index=batch_index,
        )
        parsed = self._try_json_load(content)
        return self._normalize_extraction_item(parsed, report_id=report_id, description=description)

    def _extract_structured_batch(self, reports_batch: List[Tuple[str, str]], batch_index: int) -> Dict[str, Dict[str, str]]:
        reports_block = self._build_batch_reports_block(reports_batch)
        report_ids = [report_id for report_id, _ in reports_batch]
        for retry in range(2):
            prompt = EXTRACTION_BATCH_PROMPT.format(reports_block=reports_block)
            content, p_tokens, r_tokens = ask_gpt(
                prompt,
                model=self.model,
                max_token=self._batch_max_tokens(len(reports_batch), retry=bool(retry)),
                base_url=self.base_url,
                api_key=self.api_key,
            )
            with self.token_lock:
                self.input_tokens += p_tokens
                self.output_tokens += r_tokens
            self._record_raw_response(
                call_type="batch",
                report_ids=report_ids,
                content=content,
                prompt_tokens=p_tokens,
                output_tokens=r_tokens,
                batch_index=batch_index,
            )

            parsed_list = self._try_json_list_load(content)
            parsed_by_id: Dict[str, Dict[str, Any]] = {}
            for item in parsed_list:
                rid = self._get_field(item, "Report_ID", "id")
                if rid and rid not in parsed_by_id:
                    parsed_by_id[rid] = item

            if parsed_by_id or retry == 1:
                normalized: Dict[str, Dict[str, str]] = {}
                for report_id, description in reports_batch:
                    parsed = parsed_by_id.get(report_id)
                    if not parsed:
                        continue
                    normalized[report_id] = self._normalize_extraction_item(
                        parsed,
                        report_id=report_id,
                        description=description,
                    )
                return normalized

        return {}

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

    def _prepare_chunk(self, batch_index: int, chunk: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        batch_input: List[Tuple[str, str]] = []
        for idx, report in chunk:
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            batch_input.append((report_id, description))

        extracted_batch = self._extract_structured_batch(batch_input, batch_index=batch_index)

        records: List[Tuple[int, str, str, Dict[str, str]]] = []
        embedding_texts: List[str] = []
        for idx, report in chunk:
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            ext = extracted_batch.get(report_id) or self._extract_structured(
                report_id=report_id,
                description=description,
                batch_index=batch_index,
            )

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
        batch_size = self.batch_size

        chunks: List[List[Tuple[int, Dict[str, Any]]]] = [
            indexed_reports[start:start + batch_size]
            for start in range(0, len(indexed_reports), batch_size)
        ]

        prepared_all: List[Dict[str, Any]] = []
        if self.workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [
                    executor.submit(self._prepare_chunk, batch_index, chunk)
                    for batch_index, chunk in enumerate(chunks, start=1)
                ]
                for future in as_completed(futures):
                    prepared_all.extend(future.result())
        else:
            for batch_index, chunk in enumerate(chunks, start=1):
                prepared_all.extend(self._prepare_chunk(batch_index, chunk))

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

            op_eid = self._match_or_create_instance_dimension("Operation", op_text, v_op, threshold=0.9)
            page_eid = self._match_or_create_instance_dimension("Page", page_text, v_pg, threshold=0.9)
            elem_eid = self._match_or_create_instance_dimension("Element", elem_text, v_el, threshold=0.9)
            bug_eid = self._match_or_create_instance_dimension("Bug", bug_text, v_bg, threshold=0.9)

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

        # Sort clusters by score descending
        sorted_clusters = sorted(self.clusters, key=lambda c: c.get("cluster_score", 1.0), reverse=True)
        
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
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[int], Dict[int, float], int, int, List[Dict[str, Any]]]:
    pipeline = GraphPriorPipeline(
        model=model,
        batch_size=batch_size,
        workers=workers,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )
    return pipeline.run(reports)
