import pytest

# =========================
# KG_desc (updated) fixtures
# =========================

EXPECTED_BUG_TYPES = [
    "功能缺陷", "崩溃问题", "布局问题", "显示问题", "网络错误",
    "空白屏幕", "性能问题", "错误提示", "乱码错误", "页面跳转问题",
]

EXPECTED_BUG_SYMPTOMS = [
    "未调用", "不生效", "不支持", "重叠", "缺失", "模糊", "不显示", "卡顿", "缓慢", "耗时过长",
    "加载失败", "崩溃", "退出", "无响应", "停止运行", "错位", "遮挡", "位置错误", "提示", "弹出",
    "报错", "SQL ERROR", "404", "502", "无法连接", "跳转异常", "返回异常", "进入错误页面", "白屏",
    "黑屏", "空白", "乱码", "无法识别",
]

EXPECTED_OPERATIONS = [
    "点击", "选择", "提交", "保存", "确认", "启用", "开启", "调用", "配置", "输入",
    "打开", "进入", "查看", "滚动", "切换", "刷新", "缩放", "加载", "查询", "搜索",
    "请求", "操作", "连续点击", "调整", "旋转", "删除", "修改", "访问", "登录", "下载",
    "重定向", "启动", "导入", "导出",
]

EXPECTED_ELEMENTS = [
    "按钮", "输入框", "文本域", "复选框", "单选框", "开关", "下拉框", "菜单项", "文本", "标签",
    "图片", "图标", "卡片", "列表", "表格", "分页器", "标签页", "页面容器", "根布局", "滚动容器",
    "弹窗", "对话框", "提示框", "进度条", "加载提示器", "占位符",
]

EXPECTED_SEVERITIES = ["低", "中", "高", "致命"]

# -------------------------
# Entity existence checks
# -------------------------

@pytest.mark.parametrize("name", EXPECTED_BUG_TYPES)
def test_bug_type_exists(session, name):
    query = "MATCH (n:BugType {name: $name}) RETURN n"
    result = session.run(query, name=name).single()
    assert result is not None, f"数据库中缺少 BugType 实体: {name}"


@pytest.mark.parametrize("name", EXPECTED_BUG_SYMPTOMS)
def test_bug_symptom_exists(session, name):
    query = "MATCH (n:BugSymptom {name: $name}) RETURN n"
    result = session.run(query, name=name).single()
    assert result is not None, f"数据库中缺少 BugSymptom 实体: {name}"


@pytest.mark.parametrize("name", EXPECTED_OPERATIONS)
def test_operation_exists(session, name):
    query = "MATCH (n:Operation {name: $name}) RETURN n"
    result = session.run(query, name=name).single()
    assert result is not None, f"数据库中缺少 Operation 实体: {name}"


@pytest.mark.parametrize("name", EXPECTED_ELEMENTS)
def test_element_exists(session, name):
    query = "MATCH (n:Element {name: $name}) RETURN n"
    result = session.run(query, name=name).single()
    assert result is not None, f"数据库中缺少 Element 实体: {name}"


@pytest.mark.parametrize("name", EXPECTED_SEVERITIES)
def test_severity_exists(session, name):
    query = "MATCH (n:Severity {name: $name}) RETURN n"
    result = session.run(query, name=name).single()
    assert result is not None, f"数据库中缺少 Severity 实体: {name}"


# -------------------------
# Relation sanity checks
# -------------------------

ASSOCIATED_WITH_CASES = [
    ("功能缺陷", "不生效"),
    ("显示问题", "模糊"),
    ("性能问题", "加载失败"),
    ("崩溃问题", "无响应"),
    ("网络错误", "无法连接"),
    ("空白屏幕", "白屏"),
]


@pytest.mark.parametrize("bug_type, symptom", ASSOCIATED_WITH_CASES)
def test_associated_with_exists(session, bug_type, symptom):
    query = """
    MATCH (a:BugType {name: $bug_type})
    MATCH (b:BugSymptom {name: $symptom})
    MATCH (a)-[:associatedWith]->(b)
    RETURN b
    """
    result = session.run(query, bug_type=bug_type, symptom=symptom).single()
    assert result is not None, f"关系缺失: ({bug_type})-[:associatedWith]->({symptom})"


TRIGGERED_BY_CASES = [
    ("功能缺陷", "点击"),
    ("显示问题", "刷新"),
    ("性能问题", "查询"),
    ("网络错误", "请求"),
    ("空白屏幕", "启动"),
    ("乱码错误", "导入"),
]


@pytest.mark.parametrize("bug_type, operation", TRIGGERED_BY_CASES)
def test_triggered_by_exists(session, bug_type, operation):
    query = """
    MATCH (a:BugType {name: $bug_type})
    MATCH (b:Operation {name: $operation})
    MATCH (a)-[:TriggeredBy]->(b)
    RETURN b
    """
    result = session.run(query, bug_type=bug_type, operation=operation).single()
    assert result is not None, f"关系缺失: ({bug_type})-[:TriggeredBy]->({operation})"


# -------------------------
# Coverage checks
# -------------------------

def test_triggered_by_operation_coverage(session):
    query = """
    MATCH (:BugType)-[:TriggeredBy]->(o:Operation)
    RETURN collect(DISTINCT o.name) AS ops
    """
    result = session.run(query).single()
    ops_in_relation = set(result["ops"] or [])

    missing_ops = sorted(set(EXPECTED_OPERATIONS) - ops_in_relation)
    assert not missing_ops, f"TriggeredBy 关系未覆盖以下 Operation: {missing_ops}"


def test_no_orphan_operations(session):
    query = """
    MATCH (o:Operation)
    WHERE NOT (o)--()
    RETURN o.name as name
    """
    results = [record["name"] for record in session.run(query)]
    if results:
        pytest.fail(f"发现孤立 Operation 节点（未连接任何关系）: {results}")
