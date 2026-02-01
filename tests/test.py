import pytest

# test_nodes

# 1. 验证缺陷类 (Bug) - 数据来源 
# 我们抽取几个典型的 Bug 名称进行验证
EXPECTED_BUGS = [
    "页面被遮挡", "按钮无法点击", "异常闪烁", "无法跳转", 
    "数据缺失", "计算错误", "上传失败"
]

@pytest.mark.parametrize("bug_name", EXPECTED_BUGS)
def test_bug_exists(session, bug_name):
    """验证核心缺陷实体是否存在"""
    query = "MATCH (b:Bug {name: $name}) RETURN b"
    result = session.run(query, name=bug_name).single()
    
    assert result is not None, f"数据库中缺少缺陷实体: {bug_name}"

# 2. 验证操作类 (Operation) - 数据来源 
EXPECTED_OPERATIONS = [
    "点击", "输入", "搜索", "登录", "拖拽", "切换网络"
]

@pytest.mark.parametrize("op_name", EXPECTED_OPERATIONS)
def test_operation_exists(session, op_name):
    """验证操作实体是否存在"""
    query = "MATCH (o:Operation {name: $name}) RETURN o"
    result = session.run(query, name=op_name).single()
    assert result is not None, f"数据库中缺少操作实体: {op_name}"

# 3. 验证风险等级 (Risk) - 数据来源 
def test_risk_levels(session):
    """验证风险等级是否只有'严重'和'偶尔'两类"""
    query = "MATCH (r:Risk) RETURN collect(r.name) as risks"
    result = session.run(query).single()["risks"]
    
    expected_risks = {"严重", "偶尔"}
    assert set(result) == expected_risks, f"风险等级不匹配，期望 {expected_risks}，实际 {result}"

# test_relations

# 数据来源 
# 格式: (操作, 关系类型, 组件)
RELATION_CASES = [
    ("点击", "verb_object", "按钮"),      # (搜索也点击按钮，或通用点击)
    ("点击", "verb_object", "输入框"),    # 
    ("进入", "verb_object", "弹窗"),      # 
    ("拖拽", "verb_object", "图片"),      # 
    ("导出", "verb_object", "表格"),      # 
]

@pytest.mark.parametrize("start_node, rel_type, end_node", RELATION_CASES)
def test_relationship_exists(session, start_node, rel_type, end_node):
    """验证三元组关系是否存在：(Operation)-[:REL]->(Element)"""
    
    query = f"""
    MATCH (a:Operation {{name: $start}})
    MATCH (b:Element {{name: $end}})
    MATCH (a)-[r:{rel_type}]->(b)
    RETURN r
    """
    result = session.run(query, start=start_node, end=end_node).single()
    
    assert result is not None, \
        f"关系断裂: ({start_node}) 应当 [{rel_type}] -> ({end_node})，但在图中未找到。"

# 验证孤立节点 (Sanity Check)
def test_no_orphan_operations(session):
    """检查是否有'操作'没有连接任何对象（可选的质量检查）"""
    # 比如 '切换账号' 如果没连任何元素，可能是数据漏了
    query = """
    MATCH (o:Operation)
    WHERE NOT (o)--()
    RETURN o.name as name
    """
    results = [record["name"] for record in session.run(query)]
    
    # 这是一个警告性的测试，如果列表不为空，打印出来提醒开发者
    if results:
        pytest.fail(f"发现孤立的操作节点（未连接任何元素）: {results}")