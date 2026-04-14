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
