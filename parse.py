import re
from typing import List, Any, Dict

PROMPT_LLMPRIOR = """
You are provided with a set of test reports.
Each report includes a description of the bug and the operations that trigger it.

The reports are listed below:
---
#####
---

Categorize all the reports into different bug types with fine grains.
For example, errors in display should be classified by their unique occurrences, not just as general display errors.
Note that:
1. Similar bug description with completely different operations that trigger the bug should be categorized as different bug types.
2. Completely different operations that trigger the bug with similar bug description should be categorized as different bug types.

Use a tree structure to display. Each category starts with "LEVEL <level-number>". The corresponding reports numbers should be appended in this format: " -> Report: n1, n2, ...".

Work this out in a step by step way to make sure we have the correct categorization.
""".strip()

PROMPT_SIMPLE = """
You are provided with a set of test reports.
Each report includes a description of the bug and the operations that trigger it.

The reports are listed below:
---
#####
---

Prioritize all the test reports to ensure a quickly review of diverse bug types.

Only generate the prioritized test report sequence.
""".strip()

PROMPT_DIRECT = """
You are provided with a set of test reports.
Each report includes a description of the bug and the operations that trigger it.

The reports are listed below:
---
#####
---

First, categorize all the reports into different bug types with fine grains.
For example, errors in display should be classified by their unique occurrences, not just as general display errors.
Note that:
1. Similar bug description with completely different operations that trigger the bug should be categorized as different bug types.
2. Completely different operations that trigger the bug with similar bug description should be categorized as different bug types.

Then, prioritize all the test reports, ensuring that reviewers can quickly encounter diverse bug types, if reviewing from highest to lowest priority.
For example, given report 1,2,3 (bug A), report 4,5 (bug B), report 6,7,8 (bug C), a good prioritization sequence is 1,4,6,2,5,7,3,8.

Show the result with a prioritization sequence of all the reports with each report in a line.
Make sure we have the correct categorization and the best report prioritization sequence.
""".strip()

DATASET_ROOT = "./reports"  # set dataset root path


def extract_report_num(line: str) -> int:
    """从 'Report N' 或 'Report N:' 或 '1. Report N' 等格式中提取报告编号。"""
    m = re.search(r'Report\s+(\d+)', line)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract report number from line: {line!r}")


def parse(s: str, mode: str = "default") -> str:
    """
    解析 LLM 响应，返回 'Report N:\\n' 格式的优先级序列。

    - simple / direct: LLM 直接输出序列，按出现顺序提取
    - default: 树形结构（LEVEL n → Report: ...），
               按先广度后深度的方式跨类别交叉，保证早发现多样性
    """
    if mode in ("simple", "direct"):
        seen: List[int] = []
        for m in re.finditer(r'Report\s+(\d+)', s):
            n = int(m.group(1))
            if n not in seen:
                seen.append(n)
        return "\n".join(f"Report {n}:" for n in seen)

    # default 模式：解析树形结构，按类别交叉排列
    # 收集每个叶节点类别的报告列表（按 LEVEL 行出现顺序）
    categories: List[List[int]] = []
    for line in s.splitlines():
        nums = re.findall(r'Report[:\s]+(\d[\d,\s]*)', line)
        if not nums:
            continue
        group: List[int] = []
        for chunk in nums:
            for n in re.findall(r'\d+', chunk):
                v = int(n)
                if v not in group:
                    group.append(v)
        if group:
            categories.append(group)

    # 跨类别轮询（round-robin），保证多样性优先
    seen_set: set = set()
    result: List[int] = []
    max_len = max((len(c) for c in categories), default=0)
    for i in range(max_len):
        for cat in categories:
            if i < len(cat) and cat[i] not in seen_set:
                result.append(cat[i])
                seen_set.add(cat[i])

    return "\n".join(f"Report {n}:" for n in result)


def load_reports(app: int) -> List[Dict[str, Any]]:
    reports = []
    try:
        with open(f"{DATASET_ROOT}/app{app}.csv", mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                items = line.strip().split(",")
                index, image_url, bug_type = items[0], items[-2], items[-1]
                description = ",".join(items[1:-2])
                reports.append({
                    "index": index,
                    "description": description,
                    "img_url": image_url,
                    "bug_type": bug_type,
                })
        return reports
    except FileNotFoundError as e:
        raise e


def load_reports_str(app: int) -> str:
    reports = load_reports(app=app)
    return "\n".join(f'Report {index + 1} - {report["description"]}' for index, report in enumerate(reports))


def get_prompt(app: int, mode: str = "default"):
    if mode == "default":
        return PROMPT_LLMPRIOR.replace("#####", load_reports_str(app))
    elif mode == "direct":
        return PROMPT_DIRECT.replace("#####", load_reports_str(app))
    elif mode == "simple":
        return PROMPT_SIMPLE.replace("#####", load_reports_str(app))
    else:
        raise Exception(f"Invalid mode: {mode}")
