'''
Knowledge Graph RAG Service for Bug Report Prioritization.
Provides vector similarity search and severity querying from Neo4j knowledge graph.
'''

from typing import Optional, List, Tuple
import os

from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_AUTH, GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class KGService:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        
        # Initialize Vector Store connected to the existing graph index
        # Kept for compatibility, though we primarily use direct driver calls for hybrid RAG
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            index_name="bug_text_index",
            node_label="Bug",
            text_node_properties=["name", "description"],
            embedding_node_property="embedding"
        )
        
        # Defined entities from the knowledge graph framework
        self.operations = [
            "点击", "输入", "退出", "进入", "搜索", "注册", "登录", "运行", 
            "编辑", "刷新", "清空", "拖拽", "滑动", "长按", "删除", "导入", 
            "导出", "切换网络", "切换账号"
        ]
        self.elements = [
            "按钮", "输入框", "弹窗", "图片", "文本标签", "复选框", "单选框", 
            "下拉菜单", "列表", "卡片", "表格", "分页器", "菜单", "图标"
        ]

    def close(self):
        if self.driver:
            self.driver.close()

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """Simple keyword matching for Operations and Elements in the text."""
        found_ops = [op for op in self.operations if op in text]
        found_elems = [elem for elem in self.elements if elem in text]
        return found_ops, found_elems

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context from the Knowledge Graph for RAG.
        Combines Vector Search (Semantic) with Graph Traversal (Structural).
        """
        context_list = []
        
        try:
            # 1. Extract Entities from Input
            ops, elems = self.extract_entities(query)
            
            # 2. Generate Embedding for vector search
            embedding = self.embeddings.embed_query(query)
            
            with self.driver.session() as session:
                candidates = {} # Map element_id -> {node, score, reasons}

                # A. Vector Search (Semantic Similarity)
                # Finds bugs with similar descriptions
                vector_res = session.run("""
                    CALL db.index.vector.queryNodes('bug_text_index', $k, $embedding)
                    YIELD node, score
                    RETURN node, score
                """, k=k, embedding=embedding)
                
                for record in vector_res:
                    node = record["node"]
                    # Use element_id as unique identifier
                    eid = node.element_id
                    candidates[eid] = {
                        "node": node,
                        "score": record["score"],
                        "reasons": ["Semantic Similarity"]
                    }

                # B. Structural Search: Operation -> Bug
                # Finds bugs triggered by the same operations mentioned in query
                if ops:
                    op_res = session.run("""
                        MATCH (o:Operation)
                        WHERE o.name IN $ops
                        MATCH (o)-[]->(bug:Bug)
                        RETURN bug
                    """, ops=ops)
                    for record in op_res:
                        node = record["bug"]
                        eid = node.element_id
                        if eid not in candidates:
                            candidates[eid] = {"node": node, "score": 1.0, "reasons": []}
                        candidates[eid]["reasons"].append(f"Linked to Operation")

                # C. Structural Search: Element <- Operation -> Bug
                # Finds bugs related to the same UI elements mentioned in query
                if elems:
                    elem_res = session.run("""
                        MATCH (e:Element)
                        WHERE e.name IN $elems
                        MATCH (e)<-[]-(o:Operation)-[]->(bug:Bug)
                        RETURN bug
                    """, elems=elems)
                    for record in elem_res:
                        node = record["bug"]
                        eid = node.element_id
                        if eid not in candidates:
                            candidates[eid] = {"node": node, "score": 0.9, "reasons": []}
                        candidates[eid]["reasons"].append(f"Linked to Element")

                # Sort candidates by score and take top results
                # We take slightly more than k to ensure diversity if we have many structural matches
                sorted_candidates = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)[:k+2]
                
                # Enrich with Structural Context for the final prompt
                for i, item in enumerate(sorted_candidates):
                    node = item["node"]
                    reasons = ", ".join(item["reasons"])
                    desc = node.get("description", "No description")
                    name = node.get("name", "Unknown Bug")
                    
                    # Fetch immediate context (Operation/Element) for this specific bug from Graph
                    # This helps explaining WHY this bug is relevant or what its context is
                    context_query = """
                        MATCH (bug:Bug) WHERE elementId(bug) = $id
                        OPTIONAL MATCH (op:Operation)-[]->(bug)
                        OPTIONAL MATCH (op)-[:verb_object]->(el:Element)
                        RETURN op.name as op_name, el.name as el_name
                    """
                    ctx_res = session.run(context_query, id=node.element_id)
                    structural_ctx = []
                    for r in ctx_res:
                        op = r["op_name"]
                        el = r["el_name"]
                        if op:
                            msg = f"Triggered by {op}"
                            if el:
                                msg += f" on {el}"
                            structural_ctx.append(msg)
                    
                    ctx_str = f"Similar Bug {i+1}: {desc}"
                    if structural_ctx:
                        # Deduplicate context strings
                        unique_ctx = list(set(structural_ctx))
                        ctx_str += f" (Context: {'; '.join(unique_ctx)})"
                    
                    # Optional: Include why it was matched (debug/transparency)
                    # if reasons:
                    #    ctx_str += f" [Match Type: {reasons}]"
                        
                    context_list.append(ctx_str)

            return "\n".join(context_list)
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def query_severity(self, query: str) -> int:
        """
        Query the severity of the most similar bug in the KG.
        Returns an integer severity score (default 3 if not found).
        """
        # Placeholder as requested to ignore severity for now
        return 3


# Singleton instance for easy import
kg_service: Optional[KGService] = None

def get_kg_service() -> Optional[KGService]:
    """Get or create the KGService singleton."""
    global kg_service
    if kg_service is None:
        try:
            kg_service = KGService()
        except Exception as e:
            print(f"Warning: Could not initialize KGService: {e}")
            kg_service = None
    return kg_service