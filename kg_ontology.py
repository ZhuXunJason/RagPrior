from dataclasses import dataclass
from typing import Set


@dataclass
class OntologyFeatures:
    operation_families: Set[str]
    symptom_families: Set[str]
    page_types: Set[str]
    modules: Set[str]
    element_types: Set[str]


# Static Ontology Dictionaries

OPERATION_FAMILY_MAP = {
    "login": "auth_action",
    "log in": "auth_action",
    "sign in": "auth_action",
    "register": "auth_action",
    "sign up": "auth_action",
    "logout": "auth_action",
    "log out": "auth_action",
    "click": "click_action",
    "tap": "click_action",
    "press": "click_action",
    "select": "click_action",
    "choose": "click_action",
    "scroll": "scroll_action",
    "swipe": "scroll_action",
    "drag": "scroll_action",
    "type": "input_action",
    "input": "input_action",
    "enter": "input_action",
    "fill": "input_action",
    "search": "search_action",
    "find": "search_action",
    "query": "search_action",
    "submit": "submit_action",
    "confirm": "submit_action",
    "save": "submit_action",
    "update": "edit_action",
    "edit": "edit_action",
    "modify": "edit_action",
    "delete": "delete_action",
    "remove": "delete_action",
    "clear": "delete_action",
    "cancel": "navigation_action",
    "back": "navigation_action",
    "return": "navigation_action",
    "close": "navigation_action",
    "open": "navigation_action",
    "view": "read_action",
    "read": "read_action",
    "check": "read_action",
    "play": "media_action",
    "pause": "media_action",
    "stop": "media_action",
    "download": "file_action",
    "upload": "file_action",
    "share": "share_action",
    "send": "share_action"
}


SYMPTOM_FAMILY_MAP = {
    "crash": "fatal",
    "freeze": "fatal",
    "not responding": "fatal",
    "anr": "fatal",
    "force close": "fatal",
    "error": "error_message",
    "exception": "error_message",
    "fail": "error_message",
    "unable": "error_message",
    "cannot": "error_message",
    "wrong": "incorrect_behavior",
    "incorrect": "incorrect_behavior",
    "invalid": "incorrect_behavior",
    "missing": "missing_content",
    "empty": "missing_content",
    "blank": "missing_content",
    "not show": "missing_content",
    "not display": "missing_content",
    "delay": "performance",
    "slow": "performance",
    "lag": "performance",
    "stuck": "performance",
    "loading": "performance",
    "ui": "ui_glitch",
    "layout": "ui_glitch",
    "overlap": "ui_glitch",
    "cut off": "ui_glitch",
    "truncate": "ui_glitch",
    "color": "ui_glitch",
    "font": "ui_glitch"
}


PAGE_MODULE_MAP = {
    "login": ["auth"],
    "register": ["auth"],
    "home": ["main"],
    "main": ["main"],
    "dashboard": ["main"],
    "profile": ["user"],
    "account": ["user"],
    "settings": ["settings"],
    "config": ["settings"],
    "search": ["search"],
    "result": ["search"],
    "detail": ["content"],
    "article": ["content"],
    "video": ["content"],
    "cart": ["shopping"],
    "checkout": ["shopping"],
    "payment": ["shopping"],
    "order": ["shopping"],
    "chat": ["communication"],
    "message": ["communication"],
    "notification": ["communication"]
}


ELEMENT_TYPE_MAP = {
    "button": "interactive",
    "btn": "interactive",
    "link": "interactive",
    "input": "input_field",
    "text field": "input_field",
    "textfield": "input_field",
    "search": "input_field",
    "image": "media",
    "img": "media",
    "icon": "media",
    "video": "media",
    "text": "text_content",
    "title": "text_content",
    "description": "text_content",
    "label": "text_content",
    "list": "container",
    "grid": "container",
    "menu": "container",
    "tab": "container",
    "dialog": "overlay",
    "modal": "overlay",
    "popup": "overlay",
    "alert": "overlay",
    "toast": "overlay",
    "snackbar": "overlay",
    "spinner": "indicator",
    "progress": "indicator",
    "loader": "indicator"
}


def normalize_text(text: str) -> str:
    """Normalize text into lowercase, stripped string."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def extract_operation_families(operation_text: str, description: str) -> Set[str]:
    """Map operation string and description context to pre-defined operation families."""
    families = set()
    text = normalize_text(operation_text) + " " + normalize_text(description)
    for kw, family in OPERATION_FAMILY_MAP.items():
        if kw in text:
            families.add(family)
    return families


def extract_symptom_families(bug_text: str, description: str) -> Set[str]:
    """Map bug string and description context to pre-defined symptom families."""
    families = set()
    text = normalize_text(bug_text) + " " + normalize_text(description)
    for kw, family in SYMPTOM_FAMILY_MAP.items():
        if kw in text:
            families.add(family)
    return families


def map_page_to_types_and_modules(page_text: str) -> tuple[Set[str], Set[str]]:
    """Map page string to page types and broad modules."""
    types = set()
    modules = set()
    text = normalize_text(page_text)
    for kw, mods in PAGE_MODULE_MAP.items():
        if kw in text:
            types.add(kw)
            for m in mods:
                modules.add(m)
    return types, modules


def map_element_to_types(element_text: str) -> Set[str]:
    """Map element string to element types."""
    types = set()
    text = normalize_text(element_text)
    for kw, t in ELEMENT_TYPE_MAP.items():
        if kw in text:
            types.add(t)
    return types


def build_ontology_features(operation: str, page: str, element: str, bug: str, description: str) -> OntologyFeatures:
    """Build a complete set of ontology features for a report."""
    op_fams = extract_operation_families(operation, description)
    symp_fams = extract_symptom_families(bug, description)
    page_types, modules = map_page_to_types_and_modules(page)
    elem_types = map_element_to_types(element)
    
    return OntologyFeatures(
        operation_families=op_fams,
        symptom_families=symp_fams,
        page_types=page_types,
        modules=modules,
        element_types=elem_types
    )