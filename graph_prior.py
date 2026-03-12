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


EXTRACTION_PROMPT = """
你是一个缺陷报告结构化抽取器。请严格按 JSON 输出，不要输出任何解释。

给定报告：
Report_ID: {report_id}
Text: {text}

请抽取并输出如下字段：
{{
  "Report_ID": "{report_id}",
  "Operation": "触发缺陷的操作描述，来自原文；若缺失填'未知操作'",
  "Page": "发生缺陷的具体页面实例描述，来自原文；若缺失填'未知页面'",
  "Element": "发生缺陷的具体组件实例描述，来自原文；若缺失填'未知组件'",
  "Bug": "具体缺陷现象描述，来自原文；若缺失填'未知缺陷'",
  "Severity_Scope": "必须是以下3个之一：element (局限于单一组件), page (影响整个页面), program (导致整个程序崩溃或阻断核心流程)。请根据缺陷的影响范围做出判断。"
}}

注意：Page 和 Element 不能同时为'未知页面'和'未知组件'，至少要尽可能提取其中一个有意义的信息。如果确实都无法提取，请提取与业务相关的实体作为 Element。
""".strip()


EXTRACTION_BATCH_PROMPT = """
你是一个缺陷报告结构化抽取器。请严格按 JSON 数组输出，不要输出任何解释。

给定多条报告：
{reports_block}

请为每条报告抽取并输出如下字段，最终输出必须是 JSON 数组：
[
    {{
        "Report_ID": "R_001",
        "Operation": "触发缺陷的操作描述，来自原文；若缺失填'未知操作'",
        "Page": "发生缺陷的具体页面实例描述，来自原文；若缺失填'未知页面'",
        "Element": "发生缺陷的具体组件实例描述，来自原文；若缺失填'未知组件'",
        "Bug": "具体缺陷现象描述，来自原文；若缺失填'未知缺陷'",
        "Severity_Scope": "必须是以下3个之一：element (局限于单一组件), page (影响整个页面), program (导致整个程序崩溃或阻断核心流程)。请根据缺陷的影响范围做出判断。"
    }}
]

注意：Page 和 Element 不能同时为'未知页面'和'未知组件'，至少要尽可能提取其中一个有意义的信息。如果确实都无法提取，请提取与业务相关的实体作为 Element。
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

    def _extract_structured(self, report_id: str, description: str) -> Dict[str, str]:
        prompt = EXTRACTION_PROMPT.format(report_id=report_id, text=description)
        content, p_tokens, r_tokens = ask_gpt(
            prompt,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        with self.token_lock:
            self.input_tokens += p_tokens
            self.output_tokens += r_tokens
        parsed = self._try_json_load(content)

        operation = str(parsed.get("Operation") or "未知操作").strip()
        page = str(parsed.get("Page") or "未知页面").strip()
        element = str(parsed.get("Element") or "未知组件").strip()
        bug = str(parsed.get("Bug") or description).strip()
        raw_severity = str(parsed.get("Severity_Scope") or "").strip().lower()
        if "program" in raw_severity:
            severity_scope = "program"
        elif "page" in raw_severity:
            severity_scope = "page"
        else:
            severity_scope = "element"

        return {
            "Report_ID": report_id,
            "Operation": operation,
            "Page": page,
            "Element": element,
            "Bug": bug,
            "Severity_Scope": severity_scope,
        }

    def _extract_structured_batch(self, reports_batch: List[Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
        reports_block = "\n\n".join(
            [f"Report_ID: {report_id}\nText: {text}" for report_id, text in reports_batch]
        )
        prompt = EXTRACTION_BATCH_PROMPT.format(reports_block=reports_block)
        content, p_tokens, r_tokens = ask_gpt(
            prompt,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        with self.token_lock:
            self.input_tokens += p_tokens
            self.output_tokens += r_tokens
        parsed_list = self._try_json_list_load(content)

        parsed_by_id: Dict[str, Dict[str, Any]] = {}
        for item in parsed_list:
            rid = str(item.get("Report_ID") or "").strip()
            if rid:
                parsed_by_id[rid] = item

        normalized: Dict[str, Dict[str, str]] = {}
        for report_id, description in reports_batch:
            parsed = parsed_by_id.get(report_id, {})

            operation = str(parsed.get("Operation") or "未知操作").strip()
            page = str(parsed.get("Page") or "未知页面").strip()
            element = str(parsed.get("Element") or "未知组件").strip()
            bug = str(parsed.get("Bug") or description).strip()
            
            raw_severity = str(parsed.get("Severity_Scope") or "").strip().lower()
            if "program" in raw_severity:
                severity_scope = "program"
            elif "page" in raw_severity:
                severity_scope = "page"
            else:
                severity_scope = "element"

            normalized[report_id] = {
                "Report_ID": report_id,
                "Operation": operation,
                "Page": page,
                "Element": element,
                "Bug": bug,
                "Severity_Scope": severity_scope,
            }

        return normalized

    def _init_in_memory_graph(self) -> None:
        for name in self.concept_elements:
            self.graph.add_node(f"Element:{name}", label="Element", name=name)
        for name in self.concept_operations:
            self.graph.add_node(f"Operation:{name}", label="Operation", name=name)
        for name in self.concept_bug_types:
            self.graph.add_node(f"BugType:{name}", label="BugType", name=name)
        for name in self.concept_bug_symptoms:
            self.graph.add_node(f"BugSymptom:{name}", label="BugSymptom", name=name)
        for name in self.concept_severity:
            self.graph.add_node(f"Severity:{name}", label="Severity", name=name)

        self._load_skeleton_from_neo4j()

    def _load_skeleton_from_neo4j(self) -> None:
        if not self.kg_service:
            return

        try:
            with self.kg_service.driver.session() as session:
                node_query = """
                MATCH (n)
                WHERE any(l IN labels(n) WHERE l IN ['Operation','Element','BugType','BugSymptom','Severity'])
                RETURN labels(n) as labels, n.name as name
                """
                for rec in session.run(node_query):
                    labels = rec.get("labels") or []
                    name = rec.get("name")
                    if not labels or not name:
                        continue
                    label = labels[0]
                    self.graph.add_node(f"{label}:{name}", label=label, name=name)

                rel_query = """
                MATCH (a)-[r:similar_to]->(b)
                RETURN labels(a) as al, a.name as an, labels(b) as bl, b.name as bn
                """
                for rec in session.run(rel_query):
                    al = rec.get("al") or []
                    bl = rec.get("bl") or []
                    an = rec.get("an")
                    bn = rec.get("bn")
                    if not al or not bl or not an or not bn:
                        continue
                    src = f"{al[0]}:{an}"
                    dst = f"{bl[0]}:{bn}"
                    self.graph.add_node(src, label=al[0], name=an)
                    self.graph.add_node(dst, label=bl[0], name=bn)
                    self.graph.add_edge(src, dst, rel="similar_to")
        except Exception as e:
            print(f"Warning: failed to load skeleton from Neo4j, continue with local skeleton only: {e}")

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

    def _prepare_chunk(self, chunk: List[Tuple[int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        batch_input: List[Tuple[str, str]] = []
        for idx, report in chunk:
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            batch_input.append((report_id, description))

        extracted_batch = self._extract_structured_batch(batch_input)

        records: List[Tuple[int, str, str, Dict[str, str]]] = []
        embedding_texts: List[str] = []
        for idx, report in chunk:
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            ext = extracted_batch.get(report_id) or self._extract_structured(report_id=report_id, description=description)

            records.append((idx, report_id, description, ext))
            embedding_texts.extend([
                ext["Operation"],
                ext["Page"],
                ext["Element"],
                ext["Bug"],
            ])

        vectors = get_embeddings(
            embedding_texts,
            model=self.model,
            embedding_model=self.embedding_model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
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
                    "v_op": vectors[base].tolist(),
                    "v_pg": vectors[base + 1].tolist(),
                    "v_el": vectors[base + 2].tolist(),
                    "v_bg": vectors[base + 3].tolist(),
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
                futures = [executor.submit(self._prepare_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    prepared_all.extend(future.result())
        else:
            for chunk in chunks:
                prepared_all.extend(self._prepare_chunk(chunk))

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

    def run(self, reports: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, float], int, int]:
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
        return sequence, score_map, self.input_tokens, self.output_tokens


def run_graph_prior(
    reports: List[Dict[str, Any]],
    model: str,
    batch_size: int = 16,
    workers: int = 4,
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[int], Dict[int, float], int, int]:
    pipeline = GraphPriorPipeline(
        model=model,
        batch_size=batch_size,
        workers=workers,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )
    return pipeline.run(reports)
