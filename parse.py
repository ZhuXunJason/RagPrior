import re
from typing import List, Dict


class TreeNode:
    def __init__(self, node_type: str, value: int):
        self.type: str = node_type
        self.children: List[TreeNode] = []
        self.value: int = value
        self.visit_count: int = 0
        self.alive: bool = True


def select_node(node: TreeNode) -> int:
    if not node.alive:
        return -1
    else:
        node.visit_count += 1
        if node.type == "report":
            node.alive = False
            return node.value
        else:
            selected_node = None
            for child in node.children:
                if child.alive:
                    selected_node = child
                    break
            if selected_node is None:
                node.alive = False
                return -1
            for child in node.children:
                if child.visit_count < selected_node.visit_count and child.alive:
                    selected_node = child
            return select_node(selected_node)


def check_alive(node: TreeNode):
    if node.type == "cat":
        for child in node.children:
            check_alive(child)
        node.alive = False
        for child in node.children:
            if child.alive:
                node.alive = True
                break


def generate_sequence(root: TreeNode) -> List[int]:
    sequence = []
    while root.alive:
        r = select_node(root)
        check_alive(root)
        if r != -1 and r not in sequence:
            sequence.append(r)
    return sequence


def parse(s: str, mode: str):
    if mode == "default":
        try:
            return parse_v1(s)
        except Exception:
            return parse_report_sequence(s)
    elif mode == "simple":
        try:
            return parse_v2(s)
        except:
            return parse_v3(s)
    elif mode == "direct":
        return parse_v3(s)


def parse_v1(s: str) -> str:
    lines = [line.strip() for line in s.split("\n")]
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("Now") and i < len(lines) // 2:
            start_line = i
    lines = lines[start_line:]

    curr_level = 0
    root = TreeNode("cat", 0)
    level_roots: Dict[str, TreeNode] = {"0": root}

    for line in lines:
        if line.startswith("```"):
            continue
        line = re.sub(r"^[\s#*│├└─]+", "", line)
        line = line.replace("- LEVEL", "LEVEL")
        line = line.replace("-> LEVEL", "LEVEL")
        line = line.replace("→", "->")
        line = line.strip()

        report_match = re.search(r"->\s*Reports?\s*:?\s*(.*)$", line, flags=re.IGNORECASE)
        if report_match:
            nums = [int(n) for n in re.findall(r"\d+", report_match.group(1))]
            node = TreeNode("cat", 0)
            for num in nums:
                node.children.append(TreeNode("report", num))
            level_roots[str(curr_level)].children.append(node)
        elif line.startswith("LEVEL "):
            match = re.match(r"LEVEL\s+(\d+)", line)
            if not match:
                continue
            new_level = int(match.group(1))
            node = TreeNode("cat", 0)
            if new_level == curr_level + 1:
                level_roots[str(curr_level)].children.append(node)
                level_roots[str(new_level)] = node
                curr_level = new_level
            elif new_level < curr_level:
                level_roots[str(new_level - 1)].children.append(node)
                level_roots[str(new_level)] = node
                curr_level = new_level
            elif new_level == curr_level:
                level_roots[str(curr_level - 1)].children.append(node)
                level_roots[str(curr_level)] = node
            else:
                raise Exception("Invalid format.")
    seq = generate_sequence(root)
    if not seq:
        raise Exception("Invalid format: no report sequence parsed.")
    res = ""
    for n in seq:
        res += f"Report {n}:\n"
    return res.strip()


def parse_report_sequence(s: str) -> str:
    """Extract report identifiers from common non-tree model output variants."""
    seq = []
    pattern = re.compile(
        r"\bReports?(?:\s+(?:IDs?|numbers?))?\s*#?\s*:?\s*([\d\s,，;、-]+)",
        flags=re.IGNORECASE,
    )
    for line in s.splitlines():
        for match in pattern.finditer(line):
            for value in re.findall(r"\d+", match.group(1)):
                report_num = int(value)
                if report_num not in seq:
                    seq.append(report_num)

    if not seq:
        raise Exception("Invalid format: no report sequence parsed.")
    return "\n".join(f"Report {n}:" for n in seq)


def parse_v2(s: str) -> str:
    s = s.replace("Report", "")
    s = s.replace(".", "")
    line = s.split("\n")[0]
    seq = []
    for n in line.split(","):
        n = int(n.strip())
        if n not in seq:
            seq.append(n)
    res = ""
    for n in seq:
        res += f"Report {n}:\n"
    return res.strip()


def parse_v3(s: str) -> str:
    lines = [line.strip() for line in s.split("\n")]
    start_line = -1
    end_line = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("1. Report"):
            start_line = i
            continue
        if start_line != -1:
            if line.find("Report") == -1:
                end_line = i
    if start_line == -1:
        raise Exception("Invalid format.")
    lines = lines[start_line:end_line]
    seq = []
    for line in lines:
        n = extract_report_num(line)
        if n not in seq:
            seq.append(n)
    res = ""
    for n in seq:
        res += f"Report {n}:\n"
    return res.strip()


def extract_report_num(line: str) -> int:
    left = line.find("Report") + len("Report")
    line = line[left:]
    try:
        return int(line.strip())
    except ValueError:
        if line.find(":") == -1 and line.find("(") != -1 and line.find("-") == -1:
            right = line.find("(")
        elif line.find(":") != -1 and line.find("(") == -1 and line.find("-") == -1:
            right = line.find(":")
        elif line.find(":") == -1 and line.find("(") == -1 and line.find("-") != -1:
            right = line.find("-")
        elif line.find(":") != -1 and line.find("(") != -1 and line.find("-") == -1:
            right = min(line.find("("), line.find(":"))
        elif line.find(":") == -1 and line.find("(") != -1 and line.find("-") != -1:
            right = min(line.find("("), line.find("-"))
        elif line.find(":") != -1 and line.find("(") == -1 and line.find("-") != -1:
            right = min(line.find(":"), line.find("-"))
        elif line.find(":") != -1 and line.find("(") != -1 and line.find("-") != -1:
            right = min(line.find("("), line.find("-"), line.find(":"))
        else:
            raise Exception("Invalid line format.")
        return int(line[:right].strip())
