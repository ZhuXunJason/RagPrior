"""
GraphPrior pipeline implementation.

Phase 1: LLM extraction + instance fusion with vector alignment
Phase 2: topology-based severity scoring
Phase 3: hybrid graph/vector deduplication
Phase 4: diversity-first sequence generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re
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
  "Instance_Page": "具体页面实例描述，来自原文；若缺失填'未知页面'",
  "Instance_Element": "具体组件实例描述，来自原文；若缺失填'未知组件'",
    "Instance_Bug": "具体缺陷现象，来自原文；若缺失填'未知缺陷'",
    "Concept_BugType": "必须是以下10个之一：功能缺陷, 崩溃问题, 布局问题, 显示问题, 网络错误, 空白屏幕, 性能问题, 错误提示, 乱码错误, 页面跳转问题",
    "Concept_BugSymptom": "必须是以下33个之一：未调用, 不生效, 不支持, 重叠, 缺失, 模糊, 不显示, 卡顿, 缓慢, 耗时过长, 加载失败, 崩溃, 退出, 无响应, 停止运行, 错位, 遮挡, 位置错误, 提示, 弹出, 报错, SQL ERROR, 404, 502, 无法连接, 跳转异常, 返回异常, 进入错误页面, 白屏, 黑屏, 空白, 乱码, 无法识别",
    "Concept_Element": "必须是以下26个之一：按钮, 输入框, 文本域, 复选框, 单选框, 开关, 下拉框, 菜单项, 文本, 标签, 图片, 图标, 卡片, 列表, 表格, 分页器, 标签页, 页面容器, 根布局, 滚动容器, 弹窗, 对话框, 提示框, 进度条, 加载提示器, 占位符",
    "Concept_Operation": "必须是以下34个之一：点击, 选择, 提交, 保存, 确认, 启用, 开启, 调用, 配置, 输入, 打开, 进入, 查看, 滚动, 切换, 刷新, 缩放, 加载, 查询, 搜索, 请求, 操作, 连续点击, 调整, 旋转, 删除, 修改, 访问, 登录, 下载, 重定向, 启动, 导入, 导出",
    "Concept_Severity": "必须是以下4个之一：低, 中, 高, 致命"
}}
""".strip()


EXTRACTION_BATCH_PROMPT = """
你是一个缺陷报告结构化抽取器。请严格按 JSON 数组输出，不要输出任何解释。

给定多条报告：
{reports_block}

请为每条报告抽取并输出如下字段，最终输出必须是 JSON 数组：
[
    {{
        "Report_ID": "R_001",
        "Instance_Page": "具体页面实例描述，来自原文；若缺失填'未知页面'",
        "Instance_Element": "具体组件实例描述，来自原文；若缺失填'未知组件'",
        "Instance_Bug": "具体缺陷现象，来自原文；若缺失填'未知缺陷'",
        "Concept_BugType": "必须是以下10个之一：功能缺陷, 崩溃问题, 布局问题, 显示问题, 网络错误, 空白屏幕, 性能问题, 错误提示, 乱码错误, 页面跳转问题",
        "Concept_BugSymptom": "必须是以下33个之一：未调用, 不生效, 不支持, 重叠, 缺失, 模糊, 不显示, 卡顿, 缓慢, 耗时过长, 加载失败, 崩溃, 退出, 无响应, 停止运行, 错位, 遮挡, 位置错误, 提示, 弹出, 报错, SQL ERROR, 404, 502, 无法连接, 跳转异常, 返回异常, 进入错误页面, 白屏, 黑屏, 空白, 乱码, 无法识别",
        "Concept_Element": "必须是以下26个之一：按钮, 输入框, 文本域, 复选框, 单选框, 开关, 下拉框, 菜单项, 文本, 标签, 图片, 图标, 卡片, 列表, 表格, 分页器, 标签页, 页面容器, 根布局, 滚动容器, 弹窗, 对话框, 提示框, 进度条, 加载提示器, 占位符",
        "Concept_Operation": "必须是以下34个之一：点击, 选择, 提交, 保存, 确认, 启用, 开启, 调用, 配置, 输入, 打开, 进入, 查看, 滚动, 切换, 刷新, 缩放, 加载, 查询, 搜索, 请求, 操作, 连续点击, 调整, 旋转, 删除, 修改, 访问, 登录, 下载, 重定向, 启动, 导入, 导出",
        "Concept_Severity": "必须是以下4个之一：低, 中, 高, 致命"
    }}
]
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
        self._instance_store: Dict[str, List[Dict[str, Any]]] = {
            "Instance_Page": [],
            "Instance_Element": [],
            "Instance_Bug": [],
        }
        self._instance_counter = 0

        self.concept_elements = [
            "按钮", "输入框", "文本域", "复选框", "单选框", "开关", "下拉框", "菜单项", "文本", "标签",
            "图片", "图标", "卡片", "列表", "表格", "分页器", "标签页", "页面容器", "根布局", "滚动容器",
            "弹窗", "对话框", "提示框", "进度条", "加载提示器", "占位符",
        ]
        self.concept_operations = [
            "点击", "选择", "提交", "保存", "确认", "启用", "开启", "调用", "配置", "输入",
            "打开", "进入", "查看", "滚动", "切换", "刷新", "缩放", "加载", "查询", "搜索",
            "请求", "操作", "连续点击", "调整", "旋转", "删除", "修改", "访问", "登录", "下载",
            "重定向", "启动", "导入", "导出",
        ]
        self.concept_bug_types = [
            "功能缺陷", "崩溃问题", "布局问题", "显示问题", "网络错误",
            "空白屏幕", "性能问题", "错误提示", "乱码错误", "页面跳转问题",
        ]
        self.concept_bug_symptoms = [
            "未调用", "不生效", "不支持", "重叠", "缺失", "模糊", "不显示", "卡顿", "缓慢", "耗时过长",
            "加载失败", "崩溃", "退出", "无响应", "停止运行", "错位", "遮挡", "位置错误", "提示", "弹出",
            "报错", "SQL ERROR", "404", "502", "无法连接", "跳转异常", "返回异常", "进入错误页面", "白屏",
            "黑屏", "空白", "乱码", "无法识别",
        ]
        self.concept_severity = ["低", "中", "高", "致命"]

        self._init_in_memory_graph()

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

    def _normalize_concept(self, value: str, allowed: List[str], fallback_text: str) -> str:
        if value in allowed:
            return value
        for option in allowed:
            if option in fallback_text:
                return option
        return allowed[0]

    def _extract_structured(self, report_id: str, description: str) -> Dict[str, str]:
        prompt = EXTRACTION_PROMPT.format(report_id=report_id, text=description)
        content, _, _ = ask_gpt(
            prompt,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
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
        concept_bug_type = self._normalize_concept(
            str(parsed.get("Concept_BugType") or "").strip(),
            self.concept_bug_types,
            description + instance_bug,
        )
        concept_bug_symptom = self._normalize_concept(
            str(parsed.get("Concept_BugSymptom") or "").strip(),
            self.concept_bug_symptoms,
            description + instance_bug,
        )
        concept_severity = self._normalize_concept(
            str(parsed.get("Concept_Severity") or "").strip(),
            self.concept_severity,
            description,
        )

        return {
            "Report_ID": report_id,
            "Instance_Page": instance_page,
            "Instance_Element": instance_element,
            "Instance_Bug": instance_bug,
            "Concept_BugType": concept_bug_type,
            "Concept_BugSymptom": concept_bug_symptom,
            "Concept_Element": concept_element,
            "Concept_Operation": concept_operation,
            "Concept_Severity": concept_severity,
        }

    def _extract_structured_batch(self, reports_batch: List[Tuple[str, str]]) -> Dict[str, Dict[str, str]]:
        reports_block = "\n\n".join(
            [f"Report_ID: {report_id}\nText: {text}" for report_id, text in reports_batch]
        )
        prompt = EXTRACTION_BATCH_PROMPT.format(reports_block=reports_block)
        content, _, _ = ask_gpt(
            prompt,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        parsed_list = self._try_json_list_load(content)

        parsed_by_id: Dict[str, Dict[str, Any]] = {}
        for item in parsed_list:
            rid = str(item.get("Report_ID") or "").strip()
            if rid:
                parsed_by_id[rid] = item

        normalized: Dict[str, Dict[str, str]] = {}
        for report_id, description in reports_batch:
            parsed = parsed_by_id.get(report_id, {})

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
            concept_bug_type = self._normalize_concept(
                str(parsed.get("Concept_BugType") or "").strip(),
                self.concept_bug_types,
                description + instance_bug,
            )
            concept_bug_symptom = self._normalize_concept(
                str(parsed.get("Concept_BugSymptom") or "").strip(),
                self.concept_bug_symptoms,
                description + instance_bug,
            )
            concept_severity = self._normalize_concept(
                str(parsed.get("Concept_Severity") or "").strip(),
                self.concept_severity,
                description,
            )

            normalized[report_id] = {
                "Report_ID": report_id,
                "Instance_Page": instance_page,
                "Instance_Element": instance_element,
                "Instance_Bug": instance_bug,
                "Concept_BugType": concept_bug_type,
                "Concept_BugSymptom": concept_bug_symptom,
                "Concept_Element": concept_element,
                "Concept_Operation": concept_operation,
                "Concept_Severity": concept_severity,
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

    def _match_or_create_instance(self, label: str, name: str, embedding: List[float], threshold: float = 0.9) -> str:
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
                ext["Instance_Page"],
                ext["Instance_Element"],
                ext["Instance_Bug"],
                description,
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
                    "v_page": vectors[base].tolist(),
                    "v_elem": vectors[base + 1].tolist(),
                    "v_bug": vectors[base + 2].tolist(),
                    "v_doc": vectors[base + 3].tolist(),
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

            page_text = ext["Instance_Page"]
            elem_text = ext["Instance_Element"]
            bug_text = ext["Instance_Bug"]

            v_page = item["v_page"]
            v_elem = item["v_elem"]
            v_bug = item["v_bug"]
            v_doc = item["v_doc"]

            page_eid = self._match_or_create_instance("Instance_Page", page_text, v_page, threshold=0.9)
            elem_eid = self._match_or_create_instance("Instance_Element", elem_text, v_elem, threshold=0.9)
            bug_eid = self._match_or_create_instance("Instance_Bug", bug_text, v_bug, threshold=0.9)

            report_node = f"Report:{report_id}"
            self.graph.add_node(
                report_node,
                label="Report",
                id=report_id,
                report_num=idx,
                description=description,
                embedding=v_doc,
                concept_operation=ext["Concept_Operation"],
                concept_element=ext["Concept_Element"],
                concept_bug_type=ext["Concept_BugType"],
                concept_bug_symptom=ext["Concept_BugSymptom"],
                concept_severity=ext["Concept_Severity"],
            )
            concept_element_node = f"Element:{ext['Concept_Element']}"
            concept_operation_node = f"Operation:{ext['Concept_Operation']}"
            concept_bug_type_node = f"BugType:{ext['Concept_BugType']}"
            concept_bug_symptom_node = f"BugSymptom:{ext['Concept_BugSymptom']}"
            concept_severity_node = f"Severity:{ext['Concept_Severity']}"

            self.graph.add_edge(elem_eid, page_eid, rel="LOCATED_ON")
            self.graph.add_edge(report_node, elem_eid, rel="OCCURS_ON")
            self.graph.add_edge(report_node, bug_eid, rel="MANIFESTS_AS")
            self.graph.add_edge(elem_eid, concept_element_node, rel="IS_A")
            self.graph.add_edge(report_node, concept_operation_node, rel="TRIGGERED_BY")
            self.graph.add_edge(report_node, concept_bug_type_node, rel="HAS_BUG_TYPE")
            self.graph.add_edge(report_node, concept_bug_symptom_node, rel="HAS_BUG_SYMPTOM")
            self.graph.add_edge(report_node, concept_severity_node, rel="HAS_SEVERITY")

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
        page_element_degree: Dict[str, set] = {}

        for row in extracted_rows:
            page_eid = str(row["page_eid"])
            elem_eid = str(row["element_eid"])
            page_element_degree.setdefault(page_eid, set()).add(elem_eid)

        for row in extracted_rows:
            report_num = int(row["report_num"])
            bug_text = str(row.get("Instance_Bug", ""))
            page_eid = str(row["page_eid"])

            is_page_scope = any(keyword in bug_text for keyword in PAGE_FAILURE_KEYWORDS)
            score = 1.0

            if is_page_scope:
                degree = len(page_element_degree.get(page_eid, set()))
                score = float(max(degree, 1))

            score_map[report_num] = score
            report_node = f"Report:{row['Report_ID']}"
            if self.graph.has_node(report_node):
                self.graph.nodes[report_node]["severity_score"] = score

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


def run_graph_prior(
    reports: List[Dict[str, Any]],
    model: str,
    batch_size: int = 16,
    workers: int = 4,
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[int], Dict[int, float]]:
    pipeline = GraphPriorPipeline(
        model=model,
        batch_size=batch_size,
        workers=workers,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
    )
    return pipeline.run(reports)
