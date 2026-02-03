'''
Knowledge Graph RAG Service for Bug Report Prioritization.
Provides vector similarity search and severity querying from Neo4j knowledge graph.
'''

from typing import Optional, List, Tuple, Dict, Set
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

    def close(self):
        if self.driver:
            self.driver.close()

    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """Simple keyword matching for Operations and Elements in the text."""
        found_ops = [op for op in self.operations if op in text]
        found_elems = [elem for elem in self.elements if elem in text]
        return found_ops, found_elems

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
                vector_res = session.run("""
                    CALL db.index.vector.queryNodes('bug_text_index', $k, $embedding)
                    YIELD node, score
                    RETURN node, score
                """, k=k, embedding=embedding)
                
                for record in vector_res:
                    node = record["node"]
                    eid = node.element_id
                    candidates[eid] = {
                        "node": node,
                        "score": record["score"],
                        "reasons": ["Semantic"]
                    }

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
                        eid = node.element_id
                        match_op = record["match_op"]
                        
                        # Calculate a simple score boost
                        score = 0.8
                        if match_op in ops: # Direct match
                            score = 1.0
                        else: # Expanded/Similar match
                            score = 0.9
                            
                        if eid not in candidates:
                            candidates[eid] = {"node": node, "score": score, "reasons": []}
                        elif candidates[eid]["score"] < score:
                            candidates[eid]["score"] = score # Upgrade score if structural match is strong
                            
                        candidates[eid]["reasons"].append(f"Op:{match_op}")

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
                        eid = node.element_id
                        match_elem = record["match_elem"]
                        
                        score = 0.7
                        if match_elem in elems:
                            score = 0.9
                        else:
                            score = 0.8
                            
                        if eid not in candidates:
                            candidates[eid] = {"node": node, "score": score, "reasons": []}
                        
                        candidates[eid]["reasons"].append(f"Elem:{match_elem}")

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
                    
                    category = info["category"] if info["category"] else "Unknown"
                    elements_str = ", ".join(info["elements"]) if info["elements"] else "Unknown"
                    pages_str = ", ".join(info["pages"]) if info["pages"] else "Unknown"
                    modules_str = ", ".join(info["modules"]) if info["modules"] else "Unknown"
                    ops_str = ", ".join(info["operations"]) if info["operations"] else "Unknown"
                    
                    # Construct the informative block
                    ctx_str = f"SimBug {i+1}: {desc}"
                    ctx_str += f"\n      [Info]: Type: {category} | Module: {modules_str} > {pages_str}"
                    ctx_str += f"\n      [Trigger]: {ops_str} on {elements_str}"
                    
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