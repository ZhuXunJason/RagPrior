"""
GraphPrior pipeline implementation.

Phase 1: LLM extraction + instance fusion with vector alignment
Phase 2: topology-based severity scoring
Phase 3: hybrid graph/vector deduplication
Phase 4: diversity-first sequence generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast, LiteralString
import json
import math
import re

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
  "Instance_Page": "具体页面实例描述，来自原文；若缺失填'未知页面'",
  "Instance_Element": "具体组件实例描述，来自原文；若缺失填'未知组件'",
  "Instance_Bug": "具体缺陷现象，来自原文；若缺失填'未知缺陷'",
  "Concept_Element": "必须是以下20个之一：按钮, 输入框, 文本标签, 表格, 文本域, 弹窗, 图片, 复选框, 分页器, 滑块, 菜单, 列表, 卡片, 下拉框, 标签页, 开关, 进度条, 时间选择器, 日期选择器, 提示框",
  "Concept_Operation": "必须是以下25个之一：点击, 输入, 进入, 编辑, 导入, 退出, 下载, 搜索, 刷新, 导出, 注册, 登录, 运行, 清空, 切换网络, 拖拽, 滑动, 长按, 删除, 切换账号, 双击, 右键点击, 填写, 确认, 取消"
}}
""".strip()


class GraphPriorPipeline:
    def __init__(self, model: str = "gemini-3-pro-preview"):
        self.model = model
        self.kg_service = get_kg_service(model_name=model)

        self.concept_elements = [
            "按钮", "输入框", "文本标签", "表格", "文本域", "弹窗", "图片", "复选框", "分页器", "滑块",
            "菜单", "列表", "卡片", "下拉框", "标签页", "开关", "进度条", "时间选择器", "日期选择器", "提示框",
        ]
        self.concept_operations = [
            "点击", "输入", "进入", "编辑", "导入", "退出", "下载", "搜索", "刷新", "导出",
            "注册", "登录", "运行", "清空", "切换网络", "拖拽", "滑动", "长按", "删除",
            "切换账号", "双击", "右键点击", "填写", "确认", "取消",
        ]

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
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def _normalize_concept(self, value: str, allowed: List[str], fallback_text: str) -> str:
        if value in allowed:
            return value
        for option in allowed:
            if option in fallback_text:
                return option
        return allowed[0]

    def _extract_structured(self, report_id: str, description: str) -> Dict[str, str]:
        prompt = EXTRACTION_PROMPT.format(report_id=report_id, text=description)
        content, _, _ = ask_gpt(prompt, model=self.model)
        parsed = self._try_json_load(content)

        instance_page = str(parsed.get("Instance_Page") or "未知页面").strip()
        instance_element = str(parsed.get("Instance_Element") or "未知组件").strip()
        instance_bug = str(parsed.get("Instance_Bug") or description).strip()

        concept_element = self._normalize_concept(
            str(parsed.get("Concept_Element") or "").strip(),
            self.concept_elements,
            description + instance_element,
        )
        concept_operation = self._normalize_concept(
            str(parsed.get("Concept_Operation") or "").strip(),
            self.concept_operations,
            description,
        )

        return {
            "Report_ID": report_id,
            "Instance_Page": instance_page,
            "Instance_Element": instance_element,
            "Instance_Bug": instance_bug,
            "Concept_Element": concept_element,
            "Concept_Operation": concept_operation,
        }

    def _ensure_vector_indexes(self, embedding_dim: int) -> None:
        if not self.kg_service:
            return

        create_queries = [
            """
            CREATE VECTOR INDEX instance_page_vec IF NOT EXISTS
            FOR (n:Instance_Page) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
            """,
            """
            CREATE VECTOR INDEX instance_element_vec IF NOT EXISTS
            FOR (n:Instance_Element) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
            """,
            """
            CREATE VECTOR INDEX instance_bug_vec IF NOT EXISTS
            FOR (n:Instance_Bug) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
            """,
            """
            CREATE VECTOR INDEX report_doc_vec IF NOT EXISTS
            FOR (n:Report) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}
            """,
        ]

        try:
            with self.kg_service.driver.session() as session:
                for q in create_queries:
                    session.run(cast(LiteralString, q), dim=embedding_dim)
        except Exception:
            # If server version doesn't support IF NOT EXISTS or vector index creation syntax,
            # continue with fallback MERGE-by-name behavior.
            return

    def _match_or_create_instance(self, label: str, name: str, embedding: List[float], index_name: str, threshold: float = 0.9) -> str:
        if not self.kg_service:
            return f"{label}:{name}"

        with self.kg_service.driver.session() as session:
            try:
                result = session.run(
                    cast(LiteralString, f"""
                    CALL db.index.vector.queryNodes('{index_name}', 1, $embedding)
                    YIELD node, score
                    RETURN elementId(node) as eid, score
                    """),
                    embedding=embedding,
                ).single()
                if result and result["score"] is not None and result["score"] > threshold:
                    return result["eid"]
            except Exception:
                # Index not ready or unavailable: fallback to exact-name match.
                pass

            exact_match = session.run(
                cast(LiteralString, f"MATCH (n:{label} {{name: $name}}) RETURN elementId(n) as eid LIMIT 1"),
                name=name,
            ).single()
            if exact_match:
                return exact_match["eid"]

            created = session.run(
                cast(LiteralString, f"CREATE (n:{label} {{name: $name, embedding: $embedding}}) RETURN elementId(n) as eid"),
                name=name,
                embedding=embedding,
            ).single()
            if created and created.get("eid"):
                return created["eid"]
            return f"{label}:{name}"

    def phase1_extract_and_fuse(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted_rows: List[Dict[str, Any]] = []

        for idx, report in enumerate(reports, start=1):
            report_id = f"R_{idx:03d}"
            description = report.get("description", "")
            ext = self._extract_structured(report_id=report_id, description=description)

            page_text = ext["Instance_Page"]
            elem_text = ext["Instance_Element"]
            bug_text = ext["Instance_Bug"]
            doc_text = description

            vectors = get_embeddings([page_text, elem_text, bug_text, doc_text], model=self.model)
            if len(vectors) != 4:
                raise RuntimeError("Embedding generation failed for GraphPrior phase 1.")

            v_page = vectors[0].tolist()
            v_elem = vectors[1].tolist()
            v_bug = vectors[2].tolist()
            v_doc = vectors[3].tolist()

            self._ensure_vector_indexes(embedding_dim=len(v_doc))

            page_eid = self._match_or_create_instance("Instance_Page", page_text, v_page, "instance_page_vec", threshold=0.9)
            elem_eid = self._match_or_create_instance("Instance_Element", elem_text, v_elem, "instance_element_vec", threshold=0.9)
            bug_eid = self._match_or_create_instance("Instance_Bug", bug_text, v_bug, "instance_bug_vec", threshold=0.9)

            if self.kg_service:
                with self.kg_service.driver.session() as session:
                    session.run(
                        """
                        MATCH (p) WHERE elementId(p) = $page_eid
                        MATCH (e) WHERE elementId(e) = $elem_eid
                        MATCH (b) WHERE elementId(b) = $bug_eid
                        MERGE (e)-[:LOCATED_ON]->(p)
                        MERGE (ce:Element {name: $concept_element})
                        MERGE (co:Operation {name: $concept_operation})
                        MERGE (r:Report {id: $report_id})
                        SET r.description = $description,
                            r.embedding = $doc_embedding,
                            r.concept_operation = $concept_operation,
                            r.concept_element = $concept_element
                        MERGE (r)-[:OCCURS_ON]->(e)
                        MERGE (r)-[:MANIFESTS_AS]->(b)
                        MERGE (e)-[:IS_A]->(ce)
                        MERGE (r)-[:TRIGGERED_BY]->(co)
                        """,
                        page_eid=page_eid,
                        elem_eid=elem_eid,
                        bug_eid=bug_eid,
                        report_id=report_id,
                        description=description,
                        doc_embedding=v_doc,
                        concept_element=ext["Concept_Element"],
                        concept_operation=ext["Concept_Operation"],
                    )

            extracted_rows.append(
                {
                    **ext,
                    "report_num": idx,
                    "page_eid": page_eid,
                    "element_eid": elem_eid,
                    "bug_eid": bug_eid,
                    "doc_embedding": v_doc,
                    "description": description,
                }
            )

        return extracted_rows

    def phase2_topology_scoring(self, extracted_rows: List[Dict[str, Any]]) -> Dict[int, float]:
        score_map: Dict[int, float] = {}

        for row in extracted_rows:
            report_num = int(row["report_num"])
            bug_text = str(row.get("Instance_Bug", ""))
            element_eid = row["element_eid"]

            is_page_scope = any(keyword in bug_text for keyword in PAGE_FAILURE_KEYWORDS)
            score = 1.0

            if is_page_scope and self.kg_service:
                with self.kg_service.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (e) WHERE elementId(e) = $elem_eid
                        MATCH (e)-[:LOCATED_ON]->(p:Instance_Page)
                        OPTIONAL MATCH (ie:Instance_Element)-[:LOCATED_ON]->(p)
                        RETURN count(DISTINCT ie) as degree
                        """,
                        elem_eid=element_eid,
                    ).single()
                    degree = int(result["degree"] or 0) if result else 0
                    score = float(max(degree, 1))

            score_map[report_num] = score

            if self.kg_service:
                with self.kg_service.driver.session() as session:
                    session.run(
                        "MATCH (r:Report {id: $report_id}) SET r.severity_score = $score",
                        report_id=row["Report_ID"],
                        score=score,
                    )

        return score_map

    def phase3_hybrid_dedup(self, extracted_rows: List[Dict[str, Any]], score_map: Dict[int, float]) -> Tuple[List[int], List[int]]:
        candidates = sorted(extracted_rows, key=lambda r: score_map.get(int(r["report_num"]), 1.0), reverse=True)

        clusters: List[Dict[str, Any]] = []
        unassigned = candidates[:]

        while unassigned:
            head = unassigned.pop(0)
            cluster = {"rep": head, "dups": []}

            kept = []
            for item in unassigned:
                hard_match = (
                    item["element_eid"] == head["element_eid"]
                    and item["bug_eid"] == head["bug_eid"]
                )

                soft_match = False
                if not hard_match:
                    sim = self._cosine_similarity(head["doc_embedding"], item["doc_embedding"])
                    same_op = item.get("Concept_Operation") == head.get("Concept_Operation")
                    soft_match = sim > 0.88 and same_op

                if hard_match or soft_match:
                    cluster["dups"].append(item)
                else:
                    kept.append(item)

            clusters.append(cluster)
            unassigned = kept

        representatives = sorted(
            [c["rep"] for c in clusters],
            key=lambda r: score_map.get(int(r["report_num"]), 1.0),
            reverse=True,
        )
        redundant = sorted(
            [x for c in clusters for x in c["dups"]],
            key=lambda r: score_map.get(int(r["report_num"]), 1.0),
            reverse=True,
        )

        round1 = [int(r["report_num"]) for r in representatives]
        round2 = [int(r["report_num"]) for r in redundant]
        return round1, round2

    def run(self, reports: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, float]]:
        extracted = self.phase1_extract_and_fuse(reports)
        score_map = self.phase2_topology_scoring(extracted)
        round1, round2 = self.phase3_hybrid_dedup(extracted, score_map)
        sequence = round1 + round2
        return sequence, score_map


def run_graph_prior(reports: List[Dict[str, Any]], model: str) -> Tuple[List[int], Dict[int, float]]:
    pipeline = GraphPriorPipeline(model=model)
    return pipeline.run(reports)
