'''
Knowledge Graph RAG Service for Bug Report Prioritization.
Provides vector similarity search, priority evidence retrieval, and severity querying from Neo4j knowledge graph.
'''

from typing import Optional, List, Tuple, Dict, Set, Any, cast, LiteralString
import os
import math
import re

from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, DashScopeEmbeddings

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_AUTH, GOOGLE_API_KEY, OPENAI_API_KEY, KG_MODE, KG_DESC_PATH
from offline_kg import OfflineKGService

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
        self._report_cache: Optional[List[Dict[str, Any]]] = None
        self._report_vectors: Optional[List[List[float]]] = None
        
        # Defined entities from the knowledge graph framework (Updated per KG_desc.md)
        self.operations = [
            "点击", "输入", "进入", "编辑", "导入", "退出", "下载", "搜索", "刷新", "导出", 
            "注册", "登录", "运行", "清空", "切换网络", "拖拽", "滑动", "长按", "删除", 
            "切换账号", "双击", "右键点击", "填写", "确认", "取消"
        ]
        self.elements = []
        self.bug_symptoms = []

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
        """Simple keyword matching for Operations only; elements are instance-only in GraphPrior."""
        found_ops = [op for op in self.operations if op in text]
        return found_ops, []

    def extract_bug_symptoms(self, text: str) -> List[str]:
        """GraphPrior no longer relies on predefined BugSymptom nodes."""
        return []

    @staticmethod
    def _collect_tokens(text: str) -> Set[str]:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text or "")
        tokens = {token for token in normalized.split() if len(token) >= 2}
        return tokens

    def _ensure_report_cache(self) -> List[Dict[str, Any]]:
        if self._report_cache is not None and self._report_vectors is not None:
            return self._report_cache

        query = """
            MATCH (r:Report)
            OPTIONAL MATCH (r)-[:TRIGGERED_BY]->(o:Operation)
            OPTIONAL MATCH (r)-[:OCCURS_ON]->(ie:Instance_Element)
            OPTIONAL MATCH (ie)-[:LOCATED_ON]->(p:Instance_Page)
            OPTIONAL MATCH (r)-[:HAS_SEVERITY]->(sev:Severity)
            RETURN
                elementId(r) AS report_eid,
                r.description AS description,
                sev.name AS severity,
                collect(DISTINCT o.name) AS operations,
                collect(DISTINCT ie.name) AS element_instances,
                collect(DISTINCT p.name) AS page_instances
        """

        records: List[Dict[str, Any]] = []
        try:
            with self.driver.session() as session:
                for record in session.run(cast(LiteralString, query)):
                    description = self._normalize_text(str(record["description"] or ""))
                    if not description:
                        continue
                    records.append(
                        {
                            "report_eid": str(record["report_eid"]),
                            "description": description,
                            "severity": str(record["severity"] or ""),
                            "operations": [item for item in list(record["operations"] or []) if item],
                            "element_instances": [item for item in list(record["element_instances"] or []) if item],
                            "page_instances": [item for item in list(record["page_instances"] or []) if item],
                            "tokens": self._collect_tokens(description),
                        }
                    )
        except Exception as e:
            print(f"Error building report cache: {e}")
            self._report_cache = []
            self._report_vectors = []
            return self._report_cache

        self._report_cache = records
        descriptions = [item["description"] for item in records]
        self._report_vectors = self._embed_documents(descriptions) if descriptions else []
        return self._report_cache

    def _score_historical_reports(
        self,
        query_text: str,
        matched_ops: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        records = self._ensure_report_cache()
        if not records or self._report_vectors is None:
            return []

        query_norm = self._normalize_text(query_text) or "unknown"
        query_vec = cast(List[float], self.embeddings.embed_query(query_norm))
        query_tokens = self._collect_tokens(query_norm)

        scored: List[Dict[str, Any]] = []
        for record, doc_vec in zip(records, self._report_vectors):
            vector_score = max(0.0, self._cosine_similarity(query_vec, doc_vec))
            op_hits = [op for op in record["operations"] if op in matched_ops]
            token_hits = len(query_tokens & record["tokens"])
            support = float(vector_score + 0.08 * len(op_hits) + 0.01 * token_hits)
            if support <= 0:
                continue
            scored.append(
                {
                    **record,
                    "vector_score": vector_score,
                    "matched_ops": op_hits,
                    "token_hits": token_hits,
                    "support_score": support,
                }
            )

        scored.sort(
            key=lambda item: (
                float(item["support_score"]),
                float(item["vector_score"]),
                len(item["matched_ops"]),
                int(item["token_hits"]),
            ),
            reverse=True,
        )
        return scored[:top_k]

    def retrieve_link_candidates(
        self,
        query_text: str,
        top_k_op: int = 4,
        top_k_history: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve compact candidate sets for GraphPrior linking.

        Returns:
        - operations: [{"id","name","score","source"}]
        - historical_reports: [{"id","report_eid","description","severity","operations","score"}]
        """
        normalized_query = self._normalize_text(query_text)
        direct_ops, _ = self.extract_entities(normalized_query)
        historical_reports = self._score_historical_reports(
            normalized_query,
            matched_ops=direct_ops,
            top_k=max(top_k_history, top_k_op + 1),
        )

        op_support: Dict[str, Dict[str, Any]] = {}
        for op in direct_ops:
            op_support[op] = {"name": op, "score": 2.0, "source": {"direct"}}

        for history in historical_reports:
            for op in history.get("operations", []):
                item = op_support.setdefault(op, {"name": op, "score": 0.0, "source": set()})
                item["score"] += 0.6 + 0.2 * float(history.get("vector_score", 0.0))
                item["source"].add("history")

        if not op_support and self.operations:
            best_ops = []
            for op in self.operations:
                if op in normalized_query:
                    best_ops.append((op, 1.5))
                else:
                    overlap = len(self._collect_tokens(op) & self._collect_tokens(normalized_query))
                    best_ops.append((op, float(overlap)))
            best_ops.sort(key=lambda item: item[1], reverse=True)
            for op, score in best_ops[:top_k_op]:
                op_support[op] = {"name": op, "score": max(0.1, score), "source": {"fallback"}}

        operation_candidates = [
            {
                "id": f"O{idx}",
                "name": item["name"],
                "score": float(item["score"]),
                "source": ",".join(sorted(item["source"])),
            }
            for idx, item in enumerate(
                sorted(op_support.values(), key=lambda value: float(value["score"]), reverse=True)[:top_k_op],
                start=1,
            )
        ]

        history_candidates = [
            {
                "id": f"H{idx}",
                "report_eid": item["report_eid"],
                "description": item["description"],
                "severity": item["severity"],
                "operations": list(item["operations"]),
                "score": float(item["support_score"]),
            }
            for idx, item in enumerate(historical_reports[:top_k_history], start=1)
        ]
        return {
            "operations": operation_candidates,
            "historical_reports": history_candidates,
        }

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
        """Retrieve historical report evidence using operation + instance-level text overlap."""
        normalized_query = self._normalize_text(query_text)
        ops, _ = self.extract_entities(" ".join([query_text, op_text]))
        page_norm = self._normalize_text(page_text)
        elem_norm = self._normalize_text(elem_text)
        bug_norm = self._normalize_text(bug_text)

        historical_reports = self._score_historical_reports(
            normalized_query,
            matched_ops=ops,
            top_k=max(top_k * 3, 30),
        )
        if not historical_reports:
            return []

        bug_tokens = self._collect_tokens(bug_norm)
        elem_tokens = self._collect_tokens(elem_norm)
        page_tokens = self._collect_tokens(page_norm)

        evidence: List[Dict[str, Any]] = []
        for item in historical_reports:
            description = self._normalize_text(item["description"])
            description_tokens = item["tokens"]
            matched_elements = [
                name for name in item.get("element_instances", [])
                if elem_norm and (name == elem_norm or elem_norm in name or name in elem_norm)
            ]
            matched_pages = [
                name for name in item.get("page_instances", [])
                if page_norm and (name == page_norm or page_norm in name or name in page_norm)
            ]
            matched_bugs: List[str] = []
            if bug_norm:
                if bug_norm in description or description in bug_norm:
                    matched_bugs = [bug_norm]
                else:
                    overlap = sorted(bug_tokens & description_tokens)
                    if overlap:
                        matched_bugs = overlap[:3]

            hit_count = (
                len(item.get("matched_ops", []))
                + len(matched_elements)
                + len(matched_pages)
                + len(matched_bugs)
            )
            if hit_count <= 0 and float(item.get("vector_score", 0.0)) <= 0:
                continue

            evidence.append(
                {
                    "report_eid": item["report_eid"],
                    "description": description,
                    "severity": item["severity"],
                    "vector_score": float(item["vector_score"]),
                    "matched_ops": list(item.get("matched_ops", [])),
                    "matched_elements": matched_elements,
                    "matched_pages": matched_pages,
                    "matched_bugs": matched_bugs,
                    "hit_count": hit_count,
                }
            )

        evidence.sort(
            key=lambda item: (float(item["vector_score"]), int(item["hit_count"])),
            reverse=True,
        )
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
kg_service_config: Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]] = None

def get_kg_service(
    model_name: str = "gemini",
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    kg_mode: Optional[str] = None,
) -> Optional[KGService]:
    """Get or create the KGService singleton."""
    global kg_service, kg_service_config
    resolved_mode = (kg_mode or KG_MODE or "offline").lower()
    current_config = (model_name, embedding_model, base_url, api_key, resolved_mode)

    if kg_service is None or kg_service_config != current_config:
        if kg_service is not None and hasattr(kg_service, "close"):
            kg_service.close()
        try:
            if resolved_mode == "offline":
                kg_service = OfflineKGService(KG_DESC_PATH)
            else:
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
