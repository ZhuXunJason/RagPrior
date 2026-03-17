'''
Knowledge Graph RAG Service for Bug Report Prioritization.
Provides vector similarity search, priority evidence retrieval, and severity querying from Neo4j knowledge graph.
'''

from typing import Optional, List, Tuple, Dict, Set, Any, cast, LiteralString
import os
import math

from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, DashScopeEmbeddings

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_AUTH, GOOGLE_API_KEY, OPENAI_API_KEY

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class KGService:
    def __init__(
        self,
        model_name: str = "gemini",
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if not NEO4J_URI:
            raise ValueError("NEO4J_URI is not configured.")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.embedding_dim: Optional[int] = None
        self.vector_index_name = "bug_text_index"

        model_lower = (model_name or "").lower()
        if "qwen" in model_lower:
            selected_embedding_model = embedding_model or "text-embedding-v4"
            print(f"Initializing KGService with DashScope Embeddings ({selected_embedding_model}) for model {model_name}")
            effective_dashscope_key = api_key or os.getenv("DASHSCOPE_API_KEY") or OPENAI_API_KEY
            if effective_dashscope_key:
                os.environ["DASHSCOPE_API_KEY"] = effective_dashscope_key
            self.embeddings = DashScopeEmbeddings(
                model=selected_embedding_model,
                dashscope_api_key=effective_dashscope_key,
            )
        elif model_name and "gemini" not in model_lower:
            selected_embedding_model = embedding_model or "text-embedding-3-large"
            print(f"Initializing KGService with OpenAI-compatible Embeddings ({selected_embedding_model}) for model {model_name}")
            self.embeddings = OpenAIEmbeddings(
                model=selected_embedding_model,
                api_key=api_key or OPENAI_API_KEY,
                base_url=base_url,
            )
        else:
            selected_embedding_model = embedding_model or "gemini-embedding-001"
            print(f"Initializing KGService with Gemini Embeddings ({selected_embedding_model})")
            self.embeddings = GoogleGenerativeAIEmbeddings(model=selected_embedding_model)

        # Determine embedding dimension and map to a dimension-specific index name.
        try:
            probe_vec = self.embeddings.embed_query("dimension_probe")
            if probe_vec:
                self.embedding_dim = len(probe_vec)
                self.vector_index_name = f"bug_text_index_{self.embedding_dim}"
        except Exception as e:
            print(f"Warning: Could not probe embedding dimension, fallback to bug_text_index: {e}")

        # Note: vector index is queried directly in Cypher; avoid eager Neo4jVector binding here,
        # which can fail at startup when historical index dimensions differ.
        self.vector_store = None
        
        # Defined entities from the knowledge graph framework (Updated per KG_desc.md)
        self.operations = [
            "点击", "输入", "进入", "编辑", "导入", "退出", "下载", "搜索", "刷新", "导出", 
            "注册", "登录", "运行", "清空", "切换网络", "拖拽", "滑动", "长按", "删除", 
            "切换账号", "双击", "右键点击", "填写", "确认", "取消"
        ]
        self.elements = [
            "按钮", "输入框", "文本标签", "表格", "文本域", "弹窗", "图片", "复选框", 
            "分页器", "滑块", "菜单", "列表", "卡片", "下拉框", "标签页", "开关", 
            "进度条", "时间选择器", "日期选择器", "提示框"
        ]
        self.bug_symptoms = [
            "未调用", "不生效", "不支持", "重叠", "缺失", "模糊", "不显示", "卡顿", "缓慢", "耗时过长",
            "加载失败", "崩溃", "退出", "无响应", "停止运行", "错位", "遮挡", "位置错误", "提示", "弹出",
            "报错", "SQL ERROR", "404", "502", "无法连接", "跳转异常", "返回异常", "进入错误页面", "白屏",
            "黑屏", "空白", "乱码", "无法识别",
        ]

    @staticmethod
    def _merge_candidate(candidates: Dict[str, Dict], node, score: float, reason: Optional[str] = None) -> None:
        """Merge a matched bug node into candidate map with score/reason updates."""
        eid = node.element_id

        if eid not in candidates:
            candidates[eid] = {"node": node, "score": score, "reasons": []}
        elif candidates[eid]["score"] < score:
            candidates[eid]["score"] = score

        if reason:
            candidates[eid]["reasons"].append(reason)

    @staticmethod
    def _join_or_unknown(items: Optional[List[str]]) -> str:
        return ", ".join(items) if items else "Unknown"

    def close(self):
        if self.driver:
            self.driver.close()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

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

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        normalized = [self._normalize_text(text) or "unknown" for text in texts]
        try:
            embed_documents = getattr(self.embeddings, "embed_documents", None)
            if callable(embed_documents):
                return cast(List[List[float]], embed_documents(normalized))
        except Exception:
            pass

        return [cast(List[float], self.embeddings.embed_query(text)) for text in normalized]

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """Simple keyword matching for Operations and Elements in the text."""
        found_ops = [op for op in self.operations if op in text]
        found_elems = [elem for elem in self.elements if elem in text]
        return found_ops, found_elems

    def extract_bug_symptoms(self, text: str) -> List[str]:
        """Simple keyword matching for bug symptoms in the text."""
        normalized = self._normalize_text(text)
        return [sym for sym in self.bug_symptoms if sym in normalized]

    def expand_entities(self, ops: List[str], elems: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Expand extracted entities using 'similar_to' relation in KG.
        Returns sets of expanded operations and elements.
        """
        expanded_ops = set(ops)
        expanded_elems = set(elems)
        
        if not ops and not elems:
            return expanded_ops, expanded_elems

        query = """
        MATCH (n) WHERE n.name IN $names
        MATCH (n)-[:similar_to]-(related)
        RETURN n.name as original, related.name as related, labels(related) as labels
        """
        
        all_names = list(ops) + list(elems)
        
        try:
            with self.driver.session() as session:
                result = session.run(query, names=all_names)
                for record in result:
                    related_name = record["related"]
                    labels = record["labels"]
                    
                    if "Operation" in labels:
                        expanded_ops.add(related_name)
                    elif "Element" in labels:
                        expanded_elems.add(related_name)
        except Exception as e:
            print(f"Error expanding entities: {e}")
            
        return expanded_ops, expanded_elems

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context from the Knowledge Graph for RAG.
        Combines Vector Search (Semantic) with Graph Traversal (Structural).
        """
        context_list = []
        
        try:
            # 1. Extract Entities from Input
            ops, elems = self.extract_entities(query)
            
            # 2. Expand Entities (find similar operations/elements)
            # This helps if user says "Click" but the relevant bug is indexed under "Double Click"
            exp_ops, exp_elems = self.expand_entities(ops, elems)
            
            # 3. Generate Embedding for vector search
            embedding = self.embeddings.embed_query(query)
            
            with self.driver.session() as session:
                candidates = {} # Map element_id -> {node, score, reasons}

                # A. Vector Search (Semantic Similarity)
                index_candidates = [self.vector_index_name]
                if self.vector_index_name != "bug_text_index":
                    index_candidates.append("bug_text_index")

                vector_loaded = False
                for idx_name in index_candidates:
                    try:
                        vector_query = f"""
                            CALL db.index.vector.queryNodes('{idx_name}', $k, $embedding)
                            YIELD node, score
                            RETURN node, score
                        """
                        vector_res = session.run(cast(LiteralString, vector_query), k=k, embedding=embedding)

                        for record in vector_res:
                            node = record["node"]
                            self._merge_candidate(candidates, node=node, score=record["score"], reason=f"Semantic:{idx_name}")
                        vector_loaded = True
                        break
                    except Exception:
                        continue

                if not vector_loaded:
                    print("Warning: No compatible vector index found for current embedding dimension. Continue with structural retrieval only.")

                # B. Structural Search: Operation -> Bug (using expanded ops)
                if exp_ops:
                    op_res = session.run("""
                        MATCH (o:Operation)
                        WHERE o.name IN $ops
                        MATCH (o)-[]->(bug:Bug)
                        RETURN bug, o.name as match_op
                    """, ops=list(exp_ops))
                    for record in op_res:
                        node = record["bug"]
                        match_op = record["match_op"]
                        
                        # Calculate a simple score boost
                        score = 0.8
                        if match_op in ops: # Direct match
                            score = 1.0
                        else: # Expanded/Similar match
                            score = 0.9
                            
                        self._merge_candidate(candidates, node=node, score=score, reason=f"Op:{match_op}")

                # C. Structural Search: Element <- Operation -> Bug (using expanded elems)
                if exp_elems:
                    elem_res = session.run("""
                        MATCH (e:Element)
                        WHERE e.name IN $elems
                        MATCH (e)<-[]-(o:Operation)-[]->(bug:Bug)
                        RETURN bug, e.name as match_elem
                    """, elems=list(exp_elems))
                    for record in elem_res:
                        node = record["bug"]
                        match_elem = record["match_elem"]
                        
                        score = 0.7
                        if match_elem in elems:
                            score = 0.9
                        else:
                            score = 0.8
                            
                        self._merge_candidate(candidates, node=node, score=score, reason=f"Elem:{match_elem}")

                # Sort candidates
                sorted_candidates = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[:k+2]
                
                # D. Context Enrichment (The "More Information" part)
                for i, item in enumerate(sorted_candidates):
                    node = item["node"]
                    desc = node.get("description", "No description")
                    
                    # Fetch rich hierarchical context
                    rich_ctx_query = """
                        MATCH (bug:Bug) WHERE elementId(bug) = $id
                        
                        // Category info
                        OPTIONAL MATCH (bug)-[:belong_to]->(cat:Category)
                        
                        // Structural chain: Bug -> Element -> Page -> Module
                        OPTIONAL MATCH (bug)-[:occur_on]->(elem:Element)
                        OPTIONAL MATCH (elem)-[:BELONGS_TO]->(page:Page)
                        OPTIONAL MATCH (page)-[:BELONGS_TO]->(mod:Module)
                        
                        // Trigger info
                        OPTIONAL MATCH (op:Operation)-[:lead]->(bug)
                        
                        RETURN 
                            cat.name as category,
                            collect(DISTINCT elem.name) as elements,
                            collect(DISTINCT page.name) as pages,
                            collect(DISTINCT mod.name) as modules,
                            collect(DISTINCT op.name) as operations
                    """
                    
                    ctx_res = session.run(rich_ctx_query, id=node.element_id)
                    info = ctx_res.single()
                    if not info:
                        info = {
                            "category": None,
                            "elements": [],
                            "pages": [],
                            "modules": [],
                            "operations": [],
                        }
                    
                    category = info["category"] if info["category"] else "Unknown"
                    elements_str = self._join_or_unknown(info["elements"])
                    pages_str = self._join_or_unknown(info["pages"])
                    modules_str = self._join_or_unknown(info["modules"])
                    ops_str = self._join_or_unknown(info["operations"])
                    
                    # Construct the informative block
                    ctx_str = f"SimBug {i+1}: {desc}"
                    ctx_str += f"\n      [Info]: Type: {category} | Module: {modules_str} > {pages_str}"
                    ctx_str += f"\n      [Trigger]: {ops_str} on {elements_str}"
                    
                    context_list.append(ctx_str)

            return "\n".join(context_list)
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def retrieve_priority_evidence(
        self,
        query_text: str,
        op_text: str,
        page_text: str,
        elem_text: str,
        bug_text: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical report evidence for priority scoring.

        Returns a list of evidence dicts with:
        - vector_score
        - severity
        - matched_ops
        - matched_elements
        - matched_bug_symptoms
        - matched_pages
        - bug_types
        - description
        """
        normalized_query = self._normalize_text(query_text)
        ops, elems = self.extract_entities(" ".join([query_text, op_text, elem_text]))
        bug_symptoms = self.extract_bug_symptoms(" ".join([query_text, bug_text]))
        page_lower = self._normalize_text(page_text).lower()
        if page_lower in {"", "未知页面"}:
            page_lower = ""

        try:
            with self.driver.session() as session:
                structural_query = """
                    MATCH (r:Report)
                    OPTIONAL MATCH (r)-[:TRIGGERED_BY]->(o:Operation)
                    OPTIONAL MATCH (r)-[:OCCURS_ON]->(ie:Instance_Element)
                    OPTIONAL MATCH (ie)-[:IS_A]->(e:Element)
                    OPTIONAL MATCH (ie)-[:LOCATED_ON]->(p:Instance_Page)
                    OPTIONAL MATCH (r)-[:HAS_BUG_SYMPTOM]->(bs:BugSymptom)
                    OPTIONAL MATCH (r)-[:HAS_BUG_TYPE]->(bt:BugType)
                    OPTIONAL MATCH (r)-[:HAS_SEVERITY]->(sev:Severity)
                    WITH
                        r,
                        sev,
                        [name IN collect(DISTINCT CASE WHEN o.name IN $ops THEN o.name END) WHERE name IS NOT NULL] AS matched_ops,
                        [name IN collect(DISTINCT CASE WHEN e.name IN $elems THEN e.name END) WHERE name IS NOT NULL] AS matched_elements,
                        [name IN collect(DISTINCT CASE WHEN bs.name IN $bug_symptoms THEN bs.name END) WHERE name IS NOT NULL] AS matched_bug_symptoms,
                        [
                            name IN collect(
                                DISTINCT CASE
                                    WHEN $page_lower <> '' AND p.name IS NOT NULL AND
                                         (toLower(p.name) CONTAINS $page_lower OR $page_lower CONTAINS toLower(p.name))
                                    THEN p.name
                                END
                            ) WHERE name IS NOT NULL
                        ] AS matched_pages,
                        collect(DISTINCT bt.name) AS bug_types
                    WITH r, sev, matched_ops, matched_elements, matched_bug_symptoms, matched_pages, bug_types,
                         size(matched_ops) + size(matched_elements) + size(matched_bug_symptoms) + size(matched_pages) AS hit_count
                    WHERE hit_count > 0
                    RETURN
                        elementId(r) AS report_eid,
                        r.description AS description,
                        sev.name AS severity,
                        matched_ops,
                        matched_elements,
                        matched_bug_symptoms,
                        matched_pages,
                        bug_types,
                        hit_count
                    ORDER BY hit_count DESC
                    LIMIT $limit
                """
                records = list(
                    session.run(
                        structural_query,
                        ops=ops,
                        elems=elems,
                        bug_symptoms=bug_symptoms,
                        page_lower=page_lower,
                        limit=max(top_k * 3, 30),
                    )
                )
        except Exception as e:
            print(f"Error retrieving priority evidence: {e}")
            return []

        if not records:
            return []

        query_vector = cast(List[float], self.embeddings.embed_query(normalized_query or "unknown"))
        descriptions = [self._normalize_text(str(record["description"] or "")) or "unknown" for record in records]
        doc_vectors = self._embed_documents(descriptions)

        evidence: List[Dict[str, Any]] = []
        for record, doc_vec in zip(records, doc_vectors):
            vector_score = max(0.0, self._cosine_similarity(query_vector, doc_vec))
            evidence.append(
                {
                    "report_eid": record["report_eid"],
                    "description": record["description"] or "",
                    "severity": record["severity"] or "",
                    "vector_score": vector_score,
                    "matched_ops": list(record["matched_ops"] or []),
                    "matched_elements": list(record["matched_elements"] or []),
                    "matched_bug_symptoms": list(record["matched_bug_symptoms"] or []),
                    "matched_pages": list(record["matched_pages"] or []),
                    "bug_types": list(record["bug_types"] or []),
                    "hit_count": int(record["hit_count"] or 0),
                }
            )

        evidence.sort(key=lambda item: (float(item["vector_score"]), int(item["hit_count"])), reverse=True)
        return evidence[:top_k]

    def query_severity(self, query: str) -> int:
        """
        Query the severity of the most similar bug in the KG.
        Returns an integer severity score (default 3 if not found).
        """
        # Placeholder as requested to ignore severity for now
        return 3


# Singleton instance for easy import
kg_service: Optional[KGService] = None
kg_service_config: Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]] = None

def get_kg_service(
    model_name: str = "gemini",
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[KGService]:
    """Get or create the KGService singleton."""
    global kg_service, kg_service_config
    current_config = (model_name, embedding_model, base_url, api_key)

    if kg_service is None or kg_service_config != current_config:
        if kg_service is not None:
            kg_service.close()
        try:
            kg_service = KGService(
                model_name=model_name,
                embedding_model=embedding_model,
                base_url=base_url,
                api_key=api_key,
            )
            kg_service_config = current_config
        except Exception as e:
            print(f"Warning: Could not initialize KGService: {e}")
            kg_service = None
            kg_service_config = None
    return kg_service
