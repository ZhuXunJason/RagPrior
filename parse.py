"""
Parser for LLM responses in Bug Report Prioritization.
Converts LLM output tree structure to prioritized report sequence.
"""

from typing import List, Dict, Optional
import re

from prompt import load_reports, load_reports_from_file

# Try to import kg_service for severity-based sorting
try:
    from kg_service import get_kg_service
except ImportError:
    get_kg_service = None


class TreeNode:
    def __init__(self, node_type: str, value: int, severity: int = 1, description: str = ""):
        self.type: str = node_type
        self.children: List[TreeNode] = []
        self.value: int = value
        self.visit_count: int = 0
        self.alive: bool = True
        self.severity: int = severity
        self.max_severity_in_subtree: int = severity
        self.description: str = description
        self.parent: Optional[TreeNode] = None

    def update_subtree_severity(self):
        """Recursively update max_severity_in_subtree based on children."""
        if not self.children:
            self.max_severity_in_subtree = self.severity
            return

        max_sev = 0
        for child in self.children:
            child.update_subtree_severity()
            if child.max_severity_in_subtree > max_sev:
                max_sev = child.max_severity_in_subtree
        self.max_severity_in_subtree = max_sev


def select_node(node: TreeNode, use_severity: bool = False) -> int:
    if not node.alive:
        return -1
    else:
        node.visit_count += 1
        if node.type == "report":
            node.alive = False
            return node.value
        else:
            alive_children = [child for child in node.children if child.alive]
            if not alive_children:
                return -1
            
            if use_severity:
                alive_children.sort(key=lambda x: (x.visit_count, -x.max_severity_in_subtree))
            else:
                alive_children.sort(key=lambda x: x.visit_count)
            
            selected_node = alive_children[0]
            return select_node(selected_node, use_severity=use_severity)


def check_alive(node: TreeNode):
    if node.type == "cat":
        for child in node.children:
            check_alive(child)
        node.alive = False
        for child in node.children:
            if child.alive:
                node.alive = True
                break


def generate_sequence(root: TreeNode, use_severity: bool = False) -> List[int]:
    sequence = []
    if use_severity:
        root.update_subtree_severity()
    
    while root.alive:
        r = select_node(root, use_severity=use_severity)
        check_alive(root)
        if r != -1 and r not in sequence:
            sequence.append(r)
    return sequence


def find_node_by_value(node: TreeNode, value: int) -> Optional[TreeNode]:
    if node.type == "report" and node.value == value:
        return node
    for child in node.children:
        res = find_node_by_value(child, value)
        if res:
            return res
    return None


def parse(s: str, mode: str, app: int = None, use_severity: bool = False, csv_file: Optional[str] = None):
    if mode == "llmprior":
        return parse_v1(s, app, use_severity=use_severity, csv_file=csv_file)
    elif mode == "simple":
        try:
            return parse_v2(s)
        except:
            return parse_v3(s)
    elif mode == "direct":
        return parse_v3(s)
    raise Exception(f"Invalid mode: {mode}")


def format_sequence_result(sequence: List[int], categories: Optional[Dict[int, str]] = None) -> str:
    """Format report sequence into output lines expected by downstream evaluation."""
    lines: List[str] = []
    for report_num in sequence:
        if categories and report_num in categories:
            lines.append(f"Report {report_num} [{categories[report_num]}]:")
        else:
            lines.append(f"Report {report_num}:")
    return "\n".join(lines)


def parse_v1(s: str, app: int = None, use_severity: bool = False, csv_file: Optional[str] = None) -> str:
    # Load reports
    reports_data = []
    if csv_file is not None:
        try:
            reports_data = load_reports_from_file(csv_file)
        except Exception as e:
            print(f"Warning: Could not load reports from {csv_file}: {e}")
    elif app is not None:
        try:
            reports_data = load_reports(app)
        except Exception as e:
            print(f"Warning: Could not load reports for app {app}: {e}")

    # Get KG service for severity queries
    kg_service = None
    if use_severity and get_kg_service:
        kg_service = get_kg_service()

    lines = [line.strip() for line in s.split("\n")]
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("Now") and i < len(lines) // 2:
            start_line = i
    lines = lines[start_line:]

    curr_level = 0
    root = TreeNode("cat", 0, description="Root")
    level_roots: Dict[str, TreeNode] = {"0": root}

    for line in lines:
        line = line.replace("- LEVEL", "LEVEL")
        line = line.replace("-> LEVEL", "LEVEL")
        if line.find("-> Report:") == -1 and line.find("-> Report") >= 0:
            line = line.replace("-> Report", "-> Report:")
        while line.startswith("#"):
            line = line[1:]
        while line.startswith("*"):
            line = line[1:]
        line = line.strip()

        if line.find("-> Report:") != -1:
            start = line.find("-> Report:") + len("-> Report:")
            raw_nums = line[start:].strip().split(",")
            nums = []
            for n_str in raw_nums:
                match = re.search(r"\d+", n_str)
                if match:
                    nums.append(int(match.group()))
            
            report_cat_desc = ""
            if line.startswith("LEVEL "):
                match = re.search(r"LEVEL\s+\d+\s+(.*?)->", line)
                if match:
                    report_cat_desc = match.group(1).strip()

            node = TreeNode("cat", 0, description=report_cat_desc)
            for num in nums:
                sev = 1
                if use_severity and kg_service and reports_data and (num - 1) < len(reports_data):
                    desc = reports_data[num - 1]["description"]
                    sev = kg_service.query_severity(desc)
                
                report_node = TreeNode("report", num, severity=sev)
                report_node.parent = node
                node.children.append(report_node)
            
            level_roots[str(curr_level)].children.append(node)
            node.parent = level_roots[str(curr_level)]
            
        elif line.startswith("LEVEL "):
            match = re.search(r"LEVEL\s+(\d+)\s*(.*)", line)
            new_level = 0
            desc = ""
            if match:
                new_level = int(match.group(1))
                desc = match.group(2).strip()
                if line.find("-> Report:") != -1:
                    continue
            else:
                continue

            node = TreeNode("cat", 0, description=desc)
            
            if new_level == curr_level + 1:
                level_roots[str(curr_level)].children.append(node)
                node.parent = level_roots[str(curr_level)]
                level_roots[str(new_level)] = node
                curr_level = new_level
            elif new_level < curr_level:
                level_roots[str(new_level - 1)].children.append(node)
                node.parent = level_roots[str(new_level - 1)]
                level_roots[str(new_level)] = node
                curr_level = new_level
            elif new_level == curr_level:
                level_roots[str(curr_level - 1)].children.append(node)
                node.parent = level_roots[str(curr_level - 1)]
                level_roots[str(curr_level)] = node
            else:
                pass
    
    seq = generate_sequence(root, use_severity=use_severity)
    categories: Dict[int, str] = {}
    for n in seq:
        llm_category = "Unknown"
        target_node = find_node_by_value(root, n)
        if target_node and target_node.parent:
            curr = target_node.parent
            cats = []
            while curr and curr.parent:
                if curr.description:
                    cats.append(curr.description)
                curr = curr.parent
            if cats:
                llm_category = "/".join(reversed(cats))

        categories[n] = llm_category

    return format_sequence_result(seq, categories=categories)


def parse_v2(s: str) -> str:
    # Remove "Input Tokens" and "Output Tokens" lines if present
    lines = s.split("\n")
    cleaned_lines = []
    for line in lines:
        if "Input Tokens" in line or "Output Tokens" in line:
            continue
        cleaned_lines.append(line)
    
    s = "\n".join(cleaned_lines)
    
    # Try parsing as a comma-separated list first
    s_clean = s.replace("Report", "").replace(".", "").replace("\n", ",")
    
    seq = []
    raw_nums = s_clean.split(",")
    for n_str in raw_nums:
        n_str = n_str.strip()
        if not n_str:
            continue
        try:
            n = int(n_str)
            if n not in seq:
                seq.append(n)
        except ValueError:
            continue
            
    return format_sequence_result(seq)


def parse_v3(s: str) -> str:
    lines = [line.strip() for line in s.split("\n")]
    seq = []
    
    # Regex to match lines starting with "Report X" or "1. Report X"
    pattern = re.compile(r"^\s*(?:\d+\.\s*)?Report\s+\d+")
    
    for line in lines:
        if pattern.match(line):
            try:
                n = extract_report_num(line)
                if n not in seq:
                    seq.append(n)
            except:
                continue
    
    if not seq:
        raise Exception("Invalid format: No valid report lines found.")

    return format_sequence_result(seq)


def extract_report_num(line: str) -> int:
    """Extract report number from a line like 'Report 1' or '1. Report 5'."""
    match = re.search(r"Report\s*(\d+)", line)
    if match:
        return int(match.group(1))
    
    try:
        left = line.find("Report") + len("Report")
        if left < len("Report"):
            raise ValueError
        line = line[left:]
        num_str = ""
        for char in line.strip():
            if char.isdigit():
                num_str += char
            else:
                break
        if num_str:
            return int(num_str)
        else:
            raise ValueError
    except ValueError:
        raise Exception(f"Invalid line format, could not extract report number: {line}")
