"""
Prompt templates and report loading utilities for Bug Report Prioritization.
"""

from typing import List, Any, Dict, Optional

from config import REPORTS_DIR

# Try to import kg_service for RAG mode
try:
    from kg_service import get_kg_service
except ImportError:
    get_kg_service = None
    print("Warning: Could not import kg_service. RAG features will be disabled.")


PROMPT_LLMPRIOR = """
You are provided with a set of test reports.
Each report includes a description of the bug and the operations that trigger it.

The reports are listed below:
---
#####
---

Categorize all the reports into different bug types with fine grains.
You MUST include EVERY report index from the input list. Do not omit any report.

For example, errors in display should be classified by their unique occurrences, not just as general display errors.
Note that:
1. Similar bug description with completely different operations that trigger the bug should be categorized as different bug types.
2. Completely different operations that trigger the bug with similar bug description should be categorized as different bug types.

Use a tree structure to display. Each category starts with "LEVEL <level-number> <Category Description>". The corresponding reports numbers should be appended in this format: " -> Report: n1, n2, ...".

STRICT OUTPUT FORMAT EXAMPLE:
LEVEL 1 Functionality
LEVEL 2 Crash
LEVEL 3 Login Crash -> Report: 1, 3
LEVEL 2 Freeze
LEVEL 3 Search Bar Freeze -> Report: 2
LEVEL 1 UI Anomalies
LEVEL 2 Color Mismatch -> Report: 4

RULES:
- Do NOT use bullet points like "-" or "*".
- You MUST add a short, descriptive name after "LEVEL X".
- The "Report:" part must immediately follow the arrow "->".
- Ensure ALL report numbers from the input are present in the final tree.

Work this out in a step by step way to make sure we have the correct categorization, but ONLY print the final tree structure in the output.
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


def load_reports_from_file(csv_path: str) -> List[Dict[str, Any]]:
    """Load reports from a specific CSV file path.
    
    CSV format: index,description,image_url,bug_id
    The last column is treated as bug_id for evaluation metrics.
    """
    reports = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                items = line.split(",")
                if len(items) < 3:
                    continue
                index = items[0]
                image_url = items[-2]
                bug_id = items[-1].strip() if items[-1].strip() else None
                description = ",".join(items[1:-2])
                reports.append({
                    "index": index,
                    "description": description,
                    "img_url": image_url,
                    "bug_id": bug_id,
                    "bug_type": bug_id,  # For backward compatibility
                })
        return reports
    except FileNotFoundError as e:
        raise e


def load_reports(app: int) -> List[Dict[str, Any]]:
    """Load reports from default app CSV file."""
    csv_path = f"{REPORTS_DIR}/app{app}.csv"
    return load_reports_from_file(csv_path)


def load_reports_str(app: Optional[int] = None, use_rag: bool = False, csv_file: Optional[str] = None, model: str = "gemini") -> str:
    """Load reports as a string for LLM prompt.
    
    In RAG mode, each report is augmented with knowledge context from the KG.
    """
    if csv_file:
        reports = load_reports_from_file(csv_file)
    else:
        if app is None:
            raise ValueError("app must be provided when csv_file is not set")
        reports = load_reports(app=app)
    
    # Get KG service if RAG mode is enabled
    kg_service = None
    if use_rag and get_kg_service:
        kg_service = get_kg_service(model_name=model)
    
    lines = []
    for index, report in enumerate(reports):
        desc = report["description"]
        line = f"Report {index + 1} - {desc}"
        
        # RAG: Retrieve similar bugs from knowledge graph
        if use_rag and kg_service:
            context = kg_service.retrieve_context(desc, k=1)
            if context:
                # Add context with a specific marker so LLM can distinguish
                line += f"\n   [Knowledge Context]: {context.replace(chr(10), ' ')}"
        
        lines.append(line)
    return "\n".join(lines)


def get_prompt(app: Optional[int] = None, mode: str = "llmprior", csv_file: Optional[str] = None, model: str = "gemini") -> str:
    """Generate the full prompt for LLM based on mode."""
    use_rag = False
    prompt_template = PROMPT_LLMPRIOR
    
    if mode == "llmprior":
        prompt_template = PROMPT_LLMPRIOR
    elif mode == "graphprior":
        prompt_template = PROMPT_LLMPRIOR
        use_rag = True
    elif mode == "direct":
        prompt_template = PROMPT_DIRECT
    elif mode == "simple":
        prompt_template = PROMPT_SIMPLE
    else:
        raise Exception(f"Invalid mode: {mode}")
        
    return prompt_template.replace("#####", load_reports_str(app, use_rag=use_rag, csv_file=csv_file, model=model))
