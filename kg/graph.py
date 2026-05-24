"""
知识图谱核心模块
基于 GraphRAG.md 中定义的实体与关系，构建 NetworkX 有向图，
并提供缺陷严重性查询、相关实体查询、上下文生成等接口。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

# ──────────────────────────────────────────────────────────────────────────────
# 静态数据（与 GraphRAG.md 保持一致）
# ──────────────────────────────────────────────────────────────────────────────

SEVERITY_ORDER: Dict[str, int] = {"低": 1, "中": 2, "高": 3, "致命": 4}

# 根据缺陷类型的典型影响程度预设严重性
BUG_TYPE_SEVERITY: Dict[str, str] = {
    "崩溃问题":   "致命",
    "空白屏幕":   "高",
    "网络错误":   "高",
    "功能缺陷":   "高",
    "性能问题":   "中",
    "错误提示":   "中",
    "页面跳转问题": "中",
    "布局问题":   "低",
    "显示问题":   "低",
    "乱码错误":   "低",
}

ALL_BUG_TYPES: List[str] = list(BUG_TYPE_SEVERITY.keys())

ALL_SYMPTOMS: List[str] = [
    "未调用", "不生效", "不支持", "无反应", "无权限", "逻辑错误", "参数异常", "功能缺失",
    "重叠", "缺失", "模糊", "不显示", "颜色异常", "样式错乱", "图标异常", "文字变形", "渲染异常",
    "卡顿", "缓慢", "耗时过长", "加载失败", "内存占用高", "CPU 占用高", "帧率低", "响应延迟", "超时",
    "崩溃", "退出", "无响应", "停止运行", "闪退", "卡死", "进程挂掉", "强制关闭",
    "错位", "遮挡", "重叠", "位置错误", "排版错乱", "自适应失效", "边界溢出",
    "提示", "弹出", "报错", "SQL ERROR", "重复弹窗",
    "404", "502", "无法连接", "请求超时", "接口失败", "弱网异常", "丢包", "鉴权失败",
    "跳转异常", "返回异常", "无法返回", "进入错误页面", "跳转空白", "重复跳转", "跳转路由错误",
    "白屏", "黑屏", "空白", "灰屏", "页面空白", "内容镂空", "加载空白", "启动黑屏",
    "乱码", "无法识别", "字符错位", "编码异常", "方块乱码", "文案编译乱码", "导出导入乱码"
]

ALL_OPERATIONS: List[str] = [
    "点击", "选择", "提交", "保存", "确认", "启用", "开启", "调用", "配置", "输入",
    "打开", "进入", "查看", "滚动", "切换", "刷新", "缩放", "加载", "查询", "搜索",
    "请求", "操作", "连续点击", "调整", "旋转", "删除", "修改", "访问", "登录", "下载",
    "重定向", "启动", "导入", "导出", "返回", "退出", "拖拽", "长按", "滑动", "投屏",
    "解压", "压缩", "同步", "授权", "截屏", "打印", "分享", "注销", "注册", "鉴权",
    "缓存清理", "播报", "视频播放", "代理", "粘贴"
]

ALL_MODULES: List[str] = [
    "用户与账号模块", "订单与交易模块", "内容与资源模块",
    "搜索与推荐模块", "系统与设置模块", "网络与通信模块",
]

ELEMENT_CATEGORIES: Dict[str, List[str]] = {
    "交互类": ["按钮", "输入框", "文本域", "复选框", "单选框", "开关", "下拉框", "菜单项"],
    "展示类": ["文本", "标签", "图片", "图标"],
    "布局类": ["卡片", "列表", "表格", "分页器", "标签页"],
    "容器类": ["页面容器", "根布局", "滚动容器"],
    "反馈类": ["弹窗", "对话框", "提示框"],
    "数据状态类": ["进度条", "加载提示器", "占位符"],
}

ALL_PAGES: List[str] = [
    "用户交互类页面", "数据展示类页面", "操作事务类页面",
    "系统状态类页面", "导航与结构页面",
]

# 关系定义
_BUG_SYMPTOM_MAP: Dict[str, List[str]] = {
    "功能缺陷": ["未调用", "不生效", "不支持", "无反应", "无权限", "逻辑错误", "参数异常", "功能缺失"],
    "显示问题": ["重叠", "缺失", "模糊", "不显示", "颜色异常", "样式错乱", "图标异常", "文字变形", "渲染异常"],
    "性能问题": ["卡顿", "缓慢", "耗时过长", "加载失败", "内存占用高", "CPU 占用高", "帧率低", "响应延迟", "超时"],
    "崩溃问题": ["崩溃", "退出", "无响应", "停止运行", "闪退", "卡死", "进程挂掉", "强制关闭"],
    "布局问题": ["错位", "遮挡", "重叠", "位置错误", "排版错乱", "自适应失效", "边界溢出"],
    "错误提示": ["提示", "弹出", "报错", "SQL ERROR", "重复弹窗"],
    "网络错误": ["404", "502", "无法连接", "请求超时", "接口失败", "弱网异常", "丢包", "鉴权失败"],
    "页面跳转问题": ["跳转异常", "返回异常", "无法返回", "进入错误页面", "跳转空白", "重复跳转", "跳转路由错误"],
    "空白屏幕": ["白屏", "黑屏", "空白", "灰屏", "页面空白", "内容镂空", "加载空白", "启动黑屏"],
    "乱码错误": ["乱码", "无法识别", "字符错位", "编码异常", "方块乱码", "文案编译乱码", "导出导入乱码"],
}

_BUG_OPERATION_MAP: Dict[str, List[str]] = {
    "功能缺陷": ["点击", "选择", "提交", "保存", "确认", "启用", "开启", "调用", "配置", "输入", "打印", "播报", "长按", "拖拽", "粘贴"],
    "显示问题": ["打开", "进入", "查看", "滚动", "切换", "刷新", "缩放", "滑动", "截屏"],
    "性能问题": ["加载", "打开", "提交", "查询", "搜索", "刷新", "请求", "缓存清理"],
    "崩溃问题": ["打开", "点击", "切换", "返回", "退出", "操作", "连续点击"],
    "布局问题": ["打开", "切换", "调整", "滚动", "旋转", "缩放", "拖拽"],
    "错误提示": ["提交", "保存", "点击", "删除", "修改", "输入", "授权", "注册"],
    "网络错误": ["访问", "打开", "请求", "刷新", "登录", "下载", "同步", "鉴权", "代理"],
    "页面跳转问题": ["点击", "跳转异常", "返回异常", "进入", "退出", "重定向", "分享", "注销"],
    "空白屏幕": ["打开", "进入", "刷新", "跳转异常", "启动", "投屏", "视频播放"],
    "乱码错误": ["打开", "查看", "导入", "导出", "输入", "显示", "解压", "压缩"],
}

_BUG_ELEMENT_MAP: Dict[str, List[str]] = {
    "功能缺陷":    ["按钮", "输入框", "下拉框", "开关", "弹窗", "对话框"],
    "崩溃问题":    ["页面容器", "根布局", "滚动容器", "按钮"],
    "布局问题":    ["卡片", "列表", "表格", "页面容器", "文本", "图片"],
    "显示问题":    ["文本", "标签", "图片", "图标", "卡片", "列表"],
    "网络错误":    ["进度条", "加载提示器", "占位符", "文本"],
    "空白屏幕":    ["页面容器", "根布局", "进度条", "加载提示器"],
    "性能问题":    ["进度条", "加载提示器", "列表", "表格"],
    "错误提示":    ["弹窗", "对话框", "提示框", "按钮"],
    "乱码错误":    ["文本", "标签", "输入框"],
    "页面跳转问题": ["按钮", "菜单项", "页面容器"],
}

_PAGE_MODULE_MAP: Dict[str, List[str]] = {
    "用户交互类页面": ["用户与账号模块", "系统与设置模块"],
    "数据展示类页面": ["内容与资源模块", "订单与交易模块", "搜索与推荐模块"],
    "操作事务类页面": ["订单与交易模块", "内容与资源模块"],
    "系统状态类页面": ["系统与设置模块", "网络与通信模块"],
    "导航与结构页面": ["系统与设置模块", "内容与资源模块"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 知识图谱类
# ──────────────────────────────────────────────────────────────────────────────

class BugKnowledgeGraph:
    """
    基于 NetworkX DiGraph 的众测缺陷知识图谱。

    节点类型（node_type 属性）：
        BugType, BugSymptom, Operation, Module, Element, Page, Severity,
        AppPage（应用特定页面）, AppElement（应用特定元素）

    边类型（rel 属性）：
        associatedWith  BugType → BugSymptom
        TriggeredBy     BugType → Operation
        Involves        BugType → Element
        belongsTo       Page → Module
        hasSeverity     BugType → Severity
        isTypeOf        AppPage/AppElement → Page/Element（通用类型）
        appearsOn       AppElement → AppPage（出现于哪个页面）
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._build_base_graph()

    # ── 构建基础图 ────────────────────────────────────────────────────────────

    def _build_base_graph(self) -> None:
        # BugType 节点
        for bt, sev in BUG_TYPE_SEVERITY.items():
            self.G.add_node(bt, node_type="BugType",
                            severity=sev,
                            severity_weight=SEVERITY_ORDER[sev])

        # BugSymptom 节点
        for s in ALL_SYMPTOMS:
            self.G.add_node(s, node_type="BugSymptom")

        # Operation 节点
        for op in ALL_OPERATIONS:
            self.G.add_node(op, node_type="Operation")

        # Module 节点
        for m in ALL_MODULES:
            self.G.add_node(m, node_type="Module")

        # Element 节点（带 category 属性）
        for cat, elems in ELEMENT_CATEGORIES.items():
            for e in elems:
                self.G.add_node(e, node_type="Element", category=cat)

        # Page 节点
        for page in ALL_PAGES:
            self.G.add_node(page, node_type="Page")

        # Severity 节点
        for sev, w in SEVERITY_ORDER.items():
            self.G.add_node(sev, node_type="Severity", weight=w)

        # ── 边 ──────────────────────────────────────────────────────────────

        # associatedWith: BugType → BugSymptom
        for bt, syms in _BUG_SYMPTOM_MAP.items():
            for s in syms:
                self.G.add_edge(bt, s, rel="associatedWith")

        # TriggeredBy: BugType → Operation
        for bt, ops in _BUG_OPERATION_MAP.items():
            for op in ops:
                if self.G.has_node(op):
                    self.G.add_edge(bt, op, rel="TriggeredBy")

        # Involves: BugType → Element
        for bt, elems in _BUG_ELEMENT_MAP.items():
            for e in elems:
                if self.G.has_node(e):
                    self.G.add_edge(bt, e, rel="Involves")

        # belongsTo: Page → Module
        for page, mods in _PAGE_MODULE_MAP.items():
            for mod in mods:
                self.G.add_edge(page, mod, rel="belongsTo")

        # hasSeverity: BugType → Severity
        for bt, sev in BUG_TYPE_SEVERITY.items():
            self.G.add_edge(bt, sev, rel="hasSeverity")

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def get_severity(self, bug_type: str) -> Tuple[str, int]:
        """返回 (严重性标签, 严重性权重)，未知类型默认返回 ('中', 2)。"""
        sev = BUG_TYPE_SEVERITY.get(bug_type, "中")
        return sev, SEVERITY_ORDER.get(sev, 2)

    def get_related_symptoms(self, bug_type: str, top_k: int = 4) -> List[str]:
        return [
            n for n in self.G.successors(bug_type)
            if self.G.nodes[n].get("node_type") == "BugSymptom"
        ][:top_k]

    def get_related_elements(self, bug_type: str, top_k: int = 3) -> List[str]:
        return [
            n for n in self.G.successors(bug_type)
            if self.G.nodes[n].get("node_type") == "Element"
        ][:top_k]

    def get_related_operations(self, bug_type: str, top_k: int = 4) -> List[str]:
        return [
            n for n in self.G.successors(bug_type)
            if self.G.nodes[n].get("node_type") == "Operation"
        ][:top_k]

    def get_modules_for_page(self, page_type: str) -> List[str]:
        """返回页面类型对应的功能模块列表。"""
        return [
            n for n in self.G.successors(page_type)
            if self.G.nodes[n].get("node_type") == "Module"
        ]

    def get_report_context(
        self,
        bug_type: str,
        page_type: Optional[str] = None,
        module: Optional[str] = None,
        element_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        汇总一份报告相关的 KG 上下文，用于生成提示词中的结构化信息。
        """
        sev, sev_weight = self.get_severity(bug_type)
        return {
            "bug_type":      bug_type,
            "severity":      sev,
            "severity_weight": sev_weight,
            "symptoms":      self.get_related_symptoms(bug_type),
            "elements":      element_types or self.get_related_elements(bug_type),
            "operations":    self.get_related_operations(bug_type),
            "page_type":     page_type,
            "module":        module,
        }

    # ── 知识图谱扩充 ──────────────────────────────────────────────────────────

    def add_app_page(
        self,
        page_name: str,
        page_type: str,
        module: Optional[str] = None,
    ) -> None:
        """
        向图中添加应用特定页面节点，并连接到对应的通用 Page 类型和 Module。

        Args:
            page_name:  应用中具体页面名称，如 "登录页"
            page_type:  对应的通用 Page 节点，如 "用户交互类页面"
            module:     所属功能模块，如 "用户与账号模块"
        """
        if not self.G.has_node(page_name):
            self.G.add_node(page_name, node_type="AppPage")
        if page_type in self.G.nodes:
            self.G.add_edge(page_name, page_type, rel="isTypeOf")
        if module and module in self.G.nodes:
            self.G.add_edge(page_name, module, rel="belongsTo")

    def add_app_element(
        self,
        elem_name: str,
        elem_type: str,
        page_name: Optional[str] = None,
    ) -> None:
        """
        向图中添加应用特定 UI 元素节点。

        Args:
            elem_name:  元素名称，如 "登录按钮"
            elem_type:  对应的通用 Element 节点，如 "按钮"
            page_name:  该元素所在的应用页面节点
        """
        if not self.G.has_node(elem_name):
            self.G.add_node(elem_name, node_type="AppElement")
        if elem_type in self.G.nodes:
            self.G.add_edge(elem_name, elem_type, rel="isTypeOf")
        if page_name and self.G.has_node(page_name):
            self.G.add_edge(elem_name, page_name, rel="appearsOn")

    def enrich_from_extractions(
        self,
        reports: List[Dict],
    ) -> None:
        """
        Phase 1 完成后，将从报告中提取的 specific_pages / specific_elements
        批量添加为 KG 新节点，并建立与抽象节点的连接。

        每份报告的 "entities" 字段须已由 extractor.extract_all() 填充。
        添加的节点：
          - AppPage 节点  →  isTypeOf → 抽象 Page 类型
                           →  belongsTo → Module（若 module 非空）
          - AppElement 节点 → isTypeOf → （暂无抽象映射，留作后续人工补充）
                             → appearsOn → 对应 AppPage（若同一报告中出现）
        同时为每份报告建立 Report 节点并连接其提取到的实体：
          - Report_N → triggeredBy → Operation
          - Report_N → hasSymptom  → BugSymptom
          - Report_N → occursOn    → AppPage（或 page_type 抽象节点）
          - Report_N → involves    → AppElement
        """
        for r in reports:
            ents = r.get("entities", {})
            report_id = f"Report_{r['index']}"

            # 添加 Report 节点
            if not self.G.has_node(report_id):
                self.G.add_node(report_id, node_type="Report")

            # 操作边
            for op in ents.get("operations", []):
                if self.G.has_node(op):
                    self.G.add_edge(report_id, op, rel="triggeredBy")

            # 症状边
            for sym in ents.get("symptoms", []):
                if self.G.has_node(sym):
                    self.G.add_edge(report_id, sym, rel="hasSymptom")

            # 具体页面节点
            page_type = ents.get("page_type", "")
            module = ents.get("module", "")
            page_nodes = []
            for pg in ents.get("specific_pages", []):
                if pg:
                    self.add_app_page(pg, page_type, module)
                    self.G.add_edge(report_id, pg, rel="occursOn")
                    page_nodes.append(pg)
            # 如无具体页面名，直接连抽象页面类型
            if not page_nodes and page_type and self.G.has_node(page_type):
                self.G.add_edge(report_id, page_type, rel="occursOn")

            # 具体元素节点
            for elem in ents.get("specific_elements", []):
                if elem:
                    if not self.G.has_node(elem):
                        self.G.add_node(elem, node_type="AppElement")
                    self.G.add_edge(report_id, elem, rel="involves")
                    # 将元素与其出现的具体页面关联
                    for pg in page_nodes:
                        if not self.G.has_edge(elem, pg):
                            self.G.add_edge(elem, pg, rel="appearsOn")

            if not page_nodes and page_type and self.G.has_node(page_type):
                self.G.add_edge(report_id, page_type, rel="occursOn")

            kg_meta = r.get("kg_meta", {})
            image_meta = kg_meta.get("image_meta") or {}
            for elem in image_meta.get("visible_elements", []) or []:
                if elem:
                    if not self.G.has_node(elem):
                        self.G.add_node(elem, node_type="AppElement")
                    self.G.add_edge(report_id, elem, rel="involves")
                    for pg in page_nodes:
                        if not self.G.has_edge(elem, pg):
                            self.G.add_edge(elem, pg, rel="appearsOn")

    # ── 多样性计算 ────────────────────────────────────────────────────────────

    def bug_type_distance(self, bt1: str, bt2: str) -> float:
        """
        基于图结构计算两个缺陷类型之间的距离（相似性的倒数）。
        共享的邻居节点（症状/操作/元素）越多，距离越小。
        返回值 ∈ [0, 1]，0 表示完全相同。
        """
        n1 = set(self.G.successors(bt1))
        n2 = set(self.G.successors(bt2))
        if not n1 or not n2:
            return 1.0
        intersection = len(n1 & n2)
        union = len(n1 | n2)
        # Jaccard 距离
        return 1.0 - intersection / union

    def get_severity_weight(self, bug_type: str) -> int:
        """返回缺陷类型的严重性权重（致命=4，高=3，中=2，低=1）。"""
        return self.G.nodes.get(bug_type, {}).get("severity_weight", 2)

    # ── 辅助 ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        nodes = self.G.number_of_nodes()
        edges = self.G.number_of_edges()
        return f"BugKnowledgeGraph: {nodes} nodes, {edges} edges"

    # ── Graph RAG：子图文本提取 ───────────────────────────────────────────────

    def get_report_subgraph_text(self, report_index: str) -> str:
        """
        提取一份报告在 KG 中的 1~2 跳邻域，生成自然语言描述。
        这是 Graph RAG 的核心：把图结构转化为 LLM 可读的知识上下文。

        输出格式（单行，分隔符 |）：
          操作:点击,查看 | 症状:崩溃,退出 | 页面:登录页→用户与账号模块 | 元素:登录按钮,标题

        Args:
            report_index: 报告的原始 index 字符串（对应 CSV 第一列）
        """
        report_id = f"Report_{report_index}"
        if not self.G.has_node(report_id):
            return ""

        ops, syms, elems = [], [], []
        page_paths: List[str] = []       # "具体页面→模块" 路径

        for nbr in self.G.successors(report_id):
            rel      = self.G[report_id][nbr].get("rel", "")
            ntype    = self.G.nodes[nbr].get("node_type", "")

            if rel == "triggeredBy":
                ops.append(nbr)
            elif rel == "hasSymptom":
                syms.append(nbr)
            elif rel == "involves":
                elems.append(nbr)
            elif rel == "occursOn":
                # 2 跳：页面 → 模块
                modules = [
                    m for m in self.G.successors(nbr)
                    if self.G.nodes[m].get("node_type") == "Module"
                ]
                if modules:
                    page_paths.append(f"{nbr}→{modules[0]}")
                else:
                    # 页面只连接到抽象页面类型
                    abs_pages = [
                        p for p in self.G.successors(nbr)
                        if self.G.nodes[p].get("node_type") == "Page"
                    ]
                    if abs_pages:
                        page_paths.append(f"{nbr}→{abs_pages[0]}")
                    else:
                        page_paths.append(nbr)

        parts = []
        if ops:
            parts.append("操作:" + ",".join(ops[:4]))
        if syms:
            parts.append("症状:" + ",".join(syms[:4]))
        if page_paths:
            parts.append("页面:" + ",".join(page_paths[:2]))
        if elems:
            parts.append("元素:" + ",".join(elems[:3]))

        return " | ".join(parts) if parts else ""

    def get_shared_pages(self, report_index: str) -> List[str]:
        """返回该报告涉及的所有页面节点名称（AppPage 或抽象 Page）。"""
        report_id = f"Report_{report_index}"
        if not self.G.has_node(report_id):
            return []
        return [
            nbr for nbr in self.G.successors(report_id)
            if self.G[report_id][nbr].get("rel") == "occursOn"
        ]
