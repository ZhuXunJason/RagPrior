import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_prior
import offline_kg


def _make_reports(total: int):
    return [{"description": f"报告 {idx} 点击帮助 页面加载不出任何信息"} for idx in range(1, total + 1)]


def _extract_report_ids(prompt: str) -> list[str]:
    payload = prompt.split("待处理报告：", 1)[1] if "待处理报告：" in prompt else prompt
    return graph_prior.re.findall(r"\[Report (R_\d+)\]", payload)


def _build_link_block(
    report_id: str,
    op: str = "O1",
    pg: str = "P1",
    el: str = "E1",
    bug: str = "B1",
    ref: str = "H1",
    scope: str = "page",
) -> str:
    return "\n".join(
        [
            f"ID: {report_id}",
            f"OP: {op}",
            f"PG: {pg}",
            f"EL: {el}",
            f"BUG: {bug}",
            f"REF: {ref}",
            f"SCOPE: {scope}",
            "###",
        ]
    )


def _build_link_line(
    report_id: str,
    op: str = "O1",
    pg: str = "P1",
    el: str = "E1",
    bug: str = "B1",
    scope: str = "page",
) -> str:
    return f"{report_id}|{op}|{pg}|{el}|{bug}|{scope}"


def _fake_embeddings(texts, **_kwargs):
    vectors = []
    for index, text in enumerate(texts):
        normalized = str(text)
        vectors.append([float(len(normalized)), float(index + 1)])
    return np.array(vectors, dtype=float)


def _fake_embeddings_unique(texts, **_kwargs):
    size = max(1, len(texts))
    return np.eye(size, dtype=float)


class _FakeKGService:
    supports_priority_evidence = False

    def __init__(self, evidence_by_bug=None):
        self.evidence_by_bug = evidence_by_bug or {}

    def retrieve_link_candidates(self, query_text, top_k_op=4, top_k_history=0, top_k_instance=4):
        return {
            "operations": [
                {"id": "O1", "name": "点击", "score": 2.0, "source": "direct"},
                {"id": "O2", "name": "打开", "score": 1.0, "source": "history"},
            ][:top_k_op],
            "pages": [{"id": "P1", "name": "帮助页面", "score": 2.0, "source": "offline-kg"}][:top_k_instance],
            "elements": [{"id": "E1", "name": "帮助按钮", "score": 1.5, "source": "offline-kg"}][:top_k_instance],
            "bugs": [{"id": "B1", "name": "加载失败", "score": 2.0, "source": "offline-kg"}][:top_k_instance],
            "historical_reports": [],
            "operation_confidence": 2.0,
            "page_confidence": 2.0,
            "element_confidence": 1.8,
            "bug_confidence": 2.0,
            "operation_evidence": {"direct_hit": True, "relation_hit": False, "lexicon_hit": False},
            "page_evidence": {"direct_hit": True, "relation_hit": False, "lexicon_hit": False},
            "element_evidence": {"direct_hit": True, "relation_hit": True, "lexicon_hit": False},
            "bug_evidence": {"direct_hit": True, "relation_hit": True, "lexicon_hit": False},
        }

    def retrieve_priority_evidence(self, query_text, op_text, page_text, elem_text, bug_text, top_k=20):
        return list(self.evidence_by_bug.get(bug_text, []))


def _make_manual_payload(
    report_id: str,
    text: str,
    operations=None,
    pages=None,
    elements=None,
    bugs=None,
    slot_prefill=None,
    llm_slots=None,
):
    operations = operations or [{"id": "O1", "name": "点击", "score": 3.0, "source": "report", "evidence": {"report_hit": True}}]
    pages = pages or []
    elements = elements or []
    bugs = bugs or [{"id": "B1", "name": "加载失败", "score": 3.0, "source": "report", "evidence": {"report_hit": True}}]
    slot_prefill = slot_prefill or {
        "Operation": {"resolved": True, "value": operations[0]["name"], "candidate_id": operations[0]["id"], "reason": "test", "score": operations[0]["score"]},
        "Page": {"resolved": not bool(pages), "value": "未知页面" if not pages else pages[0]["name"], "candidate_id": "P0" if not pages else pages[0]["id"], "reason": "test", "score": pages[0]["score"] if pages else 0.0},
        "Element": {"resolved": not bool(elements), "value": "未知组件" if not elements else elements[0]["name"], "candidate_id": "E0" if not elements else elements[0]["id"], "reason": "test", "score": elements[0]["score"] if elements else 0.0},
        "Bug": {"resolved": True, "value": bugs[0]["name"], "candidate_id": bugs[0]["id"], "reason": "test", "score": bugs[0]["score"]},
    }
    llm_slots = llm_slots if llm_slots is not None else [slot for slot, info in slot_prefill.items() if not info.get("resolved")]
    return {
        "report_id": report_id,
        "text": text,
        "operations": operations,
        "pages": pages,
        "elements": elements,
        "bugs": bugs,
        "operation_confidence": 2.0,
        "page_confidence": 2.0,
        "element_confidence": 2.0,
        "bug_confidence": 2.0,
        "slot_prefill": slot_prefill,
        "llm_slots": llm_slots,
        "llm_required": bool(llm_slots),
        "slot_evidence": {
            "Operation": {"confidence": 2.0, "kg": {}, "candidates": {"count": len(operations), "top": operations[:3]}},
            "Page": {"confidence": 2.0, "kg": {}, "candidates": {"count": len(pages), "top": pages[:3]}},
            "Element": {"confidence": 2.0, "kg": {}, "candidates": {"count": len(elements), "top": elements[:3]}},
            "Bug": {"confidence": 2.0, "kg": {}, "candidates": {"count": len(bugs), "top": bugs[:3]}},
        },
    }


def test_batch_size_groups_requests_and_limits_tokens(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        call_log.append({"ids": report_ids, "max_token": max_token, "prompt": prompt})
        return "\n".join(_build_link_block(report_id) for report_id in report_ids), 10, 20

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=4, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "设置页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[
            {"id": "E1", "name": "帮助按钮", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "E2", "name": "帮助入口", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        bugs=[
            {"id": "B1", "name": "加载失败", "score": 1.2, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "内容不显示", "score": 1.1, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 3.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
        },
        llm_slots=["Page", "Element", "Bug"],
    )
    sequence, score_map, _, _, _ = pipeline.run(_make_reports(9))

    assert len(call_log) == 3
    assert [item["ids"] for item in call_log] == [
        ["R_001", "R_002", "R_003", "R_004"],
        ["R_005", "R_006", "R_007", "R_008"],
        ["R_009"],
    ]
    assert [item["max_token"] for item in call_log] == [1024, 1024, 256]
    assert len(sequence) == 9
    assert len(score_map) == 9


def test_link_selection_ids_are_resolved_to_instance_text(monkeypatch):
    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        return _build_link_line("R_001", op="O1", pg="P1", el="E1", bug="B1", scope="page"), 1, 1

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    payload = pipeline._build_link_candidates("R_001", "点击帮助 页面加载不出任何信息")
    result = pipeline._extract_structured_batch([payload], batch_index=1)

    assert result["R_001"]["Operation"] == "点击"
    assert "帮助" in result["R_001"]["Page"] or result["R_001"]["Page"] == "未知页面"
    assert result["R_001"]["Bug"] != "B1"


def test_partial_batch_response_only_falls_back_for_missing_reports(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        call_log.append({"ids": report_ids, "max_token": max_token})
        if len(report_ids) == 2:
            return "\n".join(
                [
                    "\n".join(
                        [
                            "ID: R_001",
                            "OP: O1",
                            "PG: P1",
                            "EL: E1",
                            "BUG: B1",
                            "SCOPE: page",
                        ]
                    ),
                    "\n".join(
                        [
                            "ID: R_002",
                            "OP: O1",
                            "PG: P1",
                            "EL: E1",
                            "BUG: B1",
                            "SCOPE: page",
                        ]
                    ),
                ]
            ), 3, 4
        return _build_link_block(report_ids[0]), 3, 4

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "设置页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[
            {"id": "E1", "name": "帮助按钮", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "E2", "name": "帮助入口", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        bugs=[
            {"id": "B1", "name": "加载失败", "score": 1.2, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "内容不显示", "score": 1.1, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 3.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
        },
        llm_slots=["Page", "Element", "Bug"],
    )
    sequence, _, _, _, _ = pipeline.run(_make_reports(2))

    assert call_log[0]["ids"] == ["R_001", "R_002"]
    assert sequence == [1, 2]
    assert len(call_log) == 1


def test_low_coverage_batch_retries_once_before_missing_fallback(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        call_log.append({"ids": report_ids, "prompt": prompt})
        if len(call_log) == 1:
            return _build_link_line("R_001"), 3, 4
        return "\n".join(_build_link_line(report_id) for report_id in report_ids), 3, 4

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "设置页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[
            {"id": "E1", "name": "帮助按钮", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "E2", "name": "帮助入口", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        bugs=[
            {"id": "B1", "name": "加载失败", "score": 1.2, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "内容不显示", "score": 1.1, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 3.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
        },
        llm_slots=["Page", "Element", "Bug"],
    )
    sequence, _, _, _, _ = pipeline.run(_make_reports(2))

    assert sequence == [1, 2]
    assert len(call_log) == 2
    assert "只按一行一个报告输出" in call_log[1]["prompt"]


def test_qwen_requests_disable_thinking(monkeypatch):
    seen_options = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        seen_options.append(request_options or {})
        report_ids = _extract_report_ids(prompt)
        return "\n".join(_build_link_line(report_id) for report_id in report_ids), 1, 1

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="qwen3.5-plus", batch_size=1, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "设置页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[],
        bugs=[
            {"id": "B1", "name": "加载失败", "score": 1.2, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "内容不显示", "score": 1.1, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 3.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": True, "value": "未知组件", "candidate_id": "E0", "reason": "no_candidates", "score": 0.0},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
        },
        llm_slots=["Page", "Bug"],
    )
    pipeline.run(_make_reports(1))

    assert seen_options
    assert seen_options[0]["extra_body"]["enable_thinking"] is False
    assert seen_options[0]["extra_body"]["thinking_budget"] == 64


def test_weak_kg_candidates_are_gated_out(monkeypatch):
    class _WeakKG(_FakeKGService):
        def retrieve_link_candidates(self, query_text, top_k_op=4, top_k_history=0, top_k_instance=4):
            return {
                "operations": [{"id": "O1", "name": "错误操作", "score": 0.2, "source": "offline-kg"}],
                "pages": [{"id": "P1", "name": "错误页面", "score": 0.2, "source": "offline-kg"}],
                "elements": [{"id": "E1", "name": "错误按钮", "score": 0.2, "source": "offline-kg"}],
                "bugs": [{"id": "B1", "name": "错误症状", "score": 0.2, "source": "offline-kg"}],
                "historical_reports": [],
                "operation_confidence": 0.2,
                "page_confidence": 0.2,
                "element_confidence": 0.2,
                "bug_confidence": 0.2,
                "operation_evidence": {"direct_hit": False, "relation_hit": False, "lexicon_hit": False},
                "page_evidence": {"direct_hit": False, "relation_hit": False, "lexicon_hit": False},
                "element_evidence": {"direct_hit": False, "relation_hit": False, "lexicon_hit": False},
                "bug_evidence": {"direct_hit": False, "relation_hit": False, "lexicon_hit": False},
            }

    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _WeakKG())
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    payload = pipeline._build_link_candidates("R_001", "点击帮助 页面加载不出任何信息")

    assert payload["operations"][0]["name"] != "错误操作"
    assert all(item["name"] != "错误页面" for item in payload["pages"])
    assert all(item["name"] != "错误按钮" for item in payload["elements"])
    assert all(item["name"] != "错误症状" for item in payload["bugs"])


def test_near_duplicate_reports_only_send_representatives_to_llm(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        call_log.append(report_ids)
        return "\n".join(_build_link_block(report_id) for report_id in report_ids), 3, 4

    def fake_embeddings(texts, **_kwargs):
        if len(texts) == 4:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.99, 0.01],
                    [0.0, 1.0],
                    [0.0, 0.99],
                ],
                dtype=float,
            )
        return _fake_embeddings(texts)

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    reports = [
        {"description": "帮助页面一直加载不出内容"},
        {"description": "帮助页面一直加载不出内容 变体"},
        {"description": "点击收藏后列表不显示"},
        {"description": "点击收藏后列表不显示 变体"},
    ]

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=4, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "收藏页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[
            {"id": "E1", "name": "帮助按钮", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "E2", "name": "收藏按钮", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        bugs=[
            {"id": "B1", "name": "加载失败", "score": 1.2, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "不显示", "score": 1.1, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 3.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
        },
        llm_slots=["Page", "Element", "Bug"],
    )
    sequence, score_map, _, _, _ = pipeline.run(reports)

    assert call_log == [["R_001", "R_003"]]
    assert len(sequence) == 4
    assert len(score_map) == 4


def test_instance_candidates_come_from_report_and_history_not_coarse_types(monkeypatch):
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)

    payload = pipeline._build_link_candidates("R_001", "点击帮助 页面加载不出任何信息")

    page_names = [item["name"] for item in payload["pages"]]
    bug_names = [item["name"] for item in payload["bugs"]]

    assert any("帮助" in name for name in page_names)
    assert any(("加载" in name) or ("信息" in name) or ("显示" in name) for name in bug_names)
    assert "用户交互类页面" not in page_names
    assert "按钮" not in page_names


def test_kg_priority_scoring_prefers_high_risk_history(monkeypatch):
    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        blocks = []
        for report_id in report_ids:
            bug_id = "B1" if report_id == "R_001" else "B2"
            blocks.append(_build_link_line(report_id, bug=bug_id))
        return "\n".join(blocks), 3, 4

    evidence_by_bug = {
        "页面加载不出任何信息": [
            {
                "vector_score": 0.9,
                "severity": "高",
                "matched_ops": ["点击"],
                "matched_elements": ["帮助"],
                "matched_pages": ["帮助页面"],
                "matched_bugs": ["加载不出任何信息"],
            }
        ],
        "普通提示": [
            {
                "vector_score": 0.8,
                "severity": "低",
                "matched_ops": ["点击"],
                "matched_elements": ["帮助"],
                "matched_pages": ["帮助页面"],
                "matched_bugs": ["提示"],
            }
        ],
    }

    class _PriorityKG(_FakeKGService):
        supports_priority_evidence = True

        def retrieve_link_candidates(self, query_text, top_k_op=4, top_k_history=0, top_k_instance=4):
            payload = super().retrieve_link_candidates(query_text, top_k_op=top_k_op, top_k_history=top_k_history, top_k_instance=top_k_instance)
            payload["bugs"] = []
            return payload

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _PriorityKG(evidence_by_bug))

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[{"id": "P1", "name": "帮助页面", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
        elements=[{"id": "E1", "name": "帮助", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
        bugs=[
            {"id": "B1", "name": "页面加载不出任何信息", "score": 1.0, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "普通提示", "score": 1.0, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 2.0},
            "Page": {"resolved": True, "value": "帮助页面", "candidate_id": "P1", "reason": "test", "score": 2.5},
            "Element": {"resolved": True, "value": "帮助", "candidate_id": "E1", "reason": "test", "score": 2.5},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.0},
        },
        llm_slots=["Bug"],
    )

    sequence, score_map, _, _, _ = pipeline.run(_make_reports(2))

    assert sequence[:2] == [1, 2]
    assert score_map[1] > score_map[2]


def test_diversity_reranking_penalizes_similar_representatives(monkeypatch):
    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        report_ids = _extract_report_ids(prompt)
        blocks = []
        for report_id in report_ids:
            if report_id == "R_001":
                blocks.append(_build_link_line(report_id, pg="P1", el="E1", bug="B1"))
            elif report_id == "R_002":
                blocks.append(_build_link_line(report_id, pg="P2", el="E2", bug="B2"))
            else:
                blocks.append(_build_link_line(report_id, pg="P3", el="E3", bug="B3"))
        return "\n".join(blocks), 3, 4

    def fake_embeddings(texts, **_kwargs):
        if texts == ["报告 1 点击帮助 页面加载不出任何信息", "报告 2 点击帮助 页面加载不出任何信息", "报告 3 点击帮助 页面加载不出任何信息"]:
            return np.eye(3, dtype=float)
        if texts == [
            "相似问题A 页面A 组件A",
            "相似问题B 页面B 组件B",
            "不同问题C 页面C 组件C",
        ]:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.95, 0.05],
                    [0.0, 1.0],
                ],
                dtype=float,
            )
        return _fake_embeddings(texts)

    class _DiversityKG(_FakeKGService):
        supports_priority_evidence = True

        def retrieve_priority_evidence(self, query_text, op_text, page_text, elem_text, bug_text, top_k=20):
            return [
                {
                    "vector_score": 0.8,
                    "severity": "高",
                    "matched_ops": ["点击"],
                    "matched_elements": ["组件"],
                    "matched_pages": ["页面"],
                    "matched_bugs": ["问题"],
                }
            ]

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _DiversityKG())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=3, workers=1)
    pipeline._build_link_candidates = lambda report_id, description: _make_manual_payload(
        report_id=report_id,
        text=description,
        pages=[
            {"id": "P1", "name": "页面A", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
            {"id": "P2", "name": "页面B", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
            {"id": "P3", "name": "页面C", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
        ],
        elements=[
            {"id": "E1", "name": "组件A", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
            {"id": "E2", "name": "组件B", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
            {"id": "E3", "name": "组件C", "score": 2.5, "source": "report", "evidence": {"report_hit": True}},
        ],
        bugs=[
            {"id": "B1", "name": "相似问题A", "score": 1.0, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B2", "name": "相似问题B", "score": 1.0, "source": "report", "evidence": {"report_hit": True}},
            {"id": "B3", "name": "不同问题C", "score": 1.0, "source": "report", "evidence": {"report_hit": True}},
        ],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 2.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.0},
            "Element": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.0},
            "Bug": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.0},
        },
        llm_slots=["Page", "Element", "Bug"],
    )

    sequence, _, _, _, _ = pipeline.run(_make_reports(3))
    assert sequence[:3] == [1, 3, 2]


def test_offline_kg_uses_operation_relations_for_candidates(tmp_path):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（2 个）
- 加载失败
- 空白

### Operation（操作行为）（2 个）
- 点击
- 打开

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（2 个）
- markers: 按钮、入口
- name: 帮助按钮 | surface: 帮助 | aliases: 帮助入口
- name: 收藏按钮 | surface: 收藏 | aliases: 收藏入口

### Page（页面实例）（2 个）
- markers: 页面、界面、页、中心
- name: 帮助页面 | surface: 帮助 | aliases: 帮助界面,帮助中心
- name: 收藏页面 | surface: 收藏 | aliases: 收藏界面

## 关系（Relations）

### OperatesOnElement（Operation → Element）
- 点击 → 帮助按钮、收藏按钮
- 打开 → 收藏按钮

### NavigatesToPage（Operation → Page）
- 点击 → 帮助页面
- 打开 → 收藏页面

### TriggersBugSymptom（Operation → BugSymptom）
- 点击 → 加载失败、空白
- 打开 → 空白
""".strip()
    path = tmp_path / "KG_desc.md"
    path.write_text(kg_text, encoding="utf-8")

    kg = offline_kg.OfflineKGService(path)
    candidates = kg.retrieve_link_candidates("点击帮助 页面加载不出任何信息", top_k_op=2, top_k_instance=3)

    assert candidates["operations"][0]["name"] == "点击"
    assert any(item["name"] == "帮助页面" for item in candidates["pages"])
    assert any(item["name"] == "帮助按钮" for item in candidates["elements"])
    assert any(item["name"] == "加载失败" for item in candidates["bugs"])
    assert candidates["page_evidence"]["relation_hit"] is True
    assert candidates["elements"][0]["evidence"]["kg_hit"] is True
    assert candidates["pages"][0]["evidence"]["matched_surface"] == "帮助"
    assert candidates["pages"][0]["evidence"]["matched_marker"] in {"界面", "中心", ""}


def test_offline_kg_legacy_page_element_entries_still_work(tmp_path):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（1 个）
- 加载失败

### Operation（操作行为）（1 个）
- 点击

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（1 个）
收藏按钮

### Page（页面实例）（2 个）
登录页
个人中心

## 关系（Relations）

### OperatesOnElement（Operation → Element）
- 点击 → 收藏按钮

### NavigatesToPage（Operation → Page）
- 点击 → 登录页、个人中心

### TriggersBugSymptom（Operation → BugSymptom）
- 点击 → 加载失败
""".strip()
    path = tmp_path / "KG_desc_legacy.md"
    path.write_text(kg_text, encoding="utf-8")

    kg = offline_kg.OfflineKGService(path)
    candidates = kg.retrieve_link_candidates("点击收藏按钮后登录界面加载失败", top_k_op=1, top_k_instance=3)

    assert any(item["name"] == "登录页" for item in candidates["pages"])
    assert any(item["name"] == "收藏按钮" for item in candidates["elements"])


def test_prefill_can_skip_llm_for_fully_resolved_report(monkeypatch):
    def fail_ask_gpt(*_args, **_kwargs):
        raise AssertionError("LLM should not be called when every slot is prefilled")

    monkeypatch.setattr(graph_prior, "ask_gpt", fail_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    payload = pipeline._build_link_candidates("R_001", "点击帮助按钮 帮助页面空白")

    assert payload["llm_required"] is False

    sequence, _, _, _, raw_responses = pipeline.run([{"description": "点击帮助按钮 帮助页面空白"}])
    assert sequence == [1]
    assert any(item["call_type"] == "prefill" for item in raw_responses)


def test_mixed_batch_only_sends_unresolved_reports_to_llm(monkeypatch):
    seen_ids = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None, request_options=None):
        ids = _extract_report_ids(prompt)
        seen_ids.extend(ids)
        return "\n".join(_build_link_line(report_id) for report_id in ids), 1, 1

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)

    def fake_payload(report_id, description):
        if report_id == "R_001":
            return _make_manual_payload(
                report_id=report_id,
                text=description,
                pages=[{"id": "P1", "name": "帮助页面", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
                elements=[{"id": "E1", "name": "帮助按钮", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
                bugs=[{"id": "B1", "name": "页面空白", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
                llm_slots=[],
            )
        return _make_manual_payload(
            report_id=report_id,
            text=description,
            pages=[
                {"id": "P1", "name": "帮助页面", "score": 1.0, "source": "lexicon", "evidence": {"lexicon_hit": True}},
                {"id": "P2", "name": "设置页面", "score": 0.95, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            ],
            elements=[],
            bugs=[{"id": "B1", "name": "加载失败", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
            slot_prefill={
                "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 2.0},
                "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.0},
                "Element": {"resolved": True, "value": "未知组件", "candidate_id": "E0", "reason": "no_candidates", "score": 0.0},
                "Bug": {"resolved": True, "value": "加载失败", "candidate_id": "B1", "reason": "test", "score": 2.5},
            },
            llm_slots=["Page"],
        )

    pipeline._build_link_candidates = fake_payload
    pipeline.run([{"description": "帮助页空白"}, {"description": "帮助页内容异常"}])

    assert seen_ids == ["R_002"]


def test_surface_matching_replaces_page_element_local_extraction(tmp_path, monkeypatch):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（1 个）
- 不显示

### Operation（操作行为）（2 个）
- 点击
- 进入

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（2 个）
- markers: 按钮、入口
- name: 帮助按钮 | surface: 帮助 | aliases: 帮助入口
- name: 保存按钮 | surface: 保存

### Page（页面实例）（2 个）
- markers: 页面、界面、页、中心
- name: 帮助页面 | surface: 帮助 | aliases: 帮助界面,帮助中心
- name: 设置页面 | surface: 设置

## 关系（Relations）

### OperatesOnElement（Operation → Element）
- 点击 → 帮助按钮、保存按钮
- 进入 → 帮助按钮

### NavigatesToPage（Operation → Page）
- 点击 → 帮助页面
- 进入 → 帮助页面

### TriggersBugSymptom（Operation → BugSymptom）
- 点击 → 不显示
- 进入 → 不显示
""".strip()
    path = tmp_path / "KG_desc_surface.md"
    path.write_text(kg_text, encoding="utf-8")

    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: offline_kg.OfflineKGService(path))
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)

    payload = pipeline._build_link_candidates("R_001", "进入帮助界面后帮助入口不显示")

    assert payload["pages"][0]["name"] == "帮助页面"
    assert payload["elements"][0]["name"] == "帮助按钮"
    assert payload["pages"][0]["evidence"]["matched_surface"] == "帮助"
    assert payload["elements"][0]["evidence"]["matched_surface"] == "帮助"
    assert payload["pages"][0]["evidence"]["relation_bonus"] >= 0.4


def test_kg_driven_local_page_element_extraction_marks_report_hits(tmp_path, monkeypatch):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（1 个）
- 不显示

### Operation（操作行为）（2 个）
- 点击
- 进入

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（2 个）
- markers: 按钮、入口
- name: 帮助按钮 | surface: 帮助 | aliases: 帮助入口
- name: 保存按钮 | surface: 保存

### Page（页面实例）（2 个）
- markers: 页面、界面、页、中心
- name: 帮助页面 | surface: 帮助 | aliases: 帮助界面,帮助中心
- name: 设置页面 | surface: 设置

## 关系（Relations）

### OperatesOnElement（Operation → Element）
- 点击 → 帮助按钮、保存按钮
- 进入 → 帮助按钮

### NavigatesToPage（Operation → Page）
- 点击 → 帮助页面
- 进入 → 帮助页面
""".strip()
    path = tmp_path / "KG_desc_local_mentions.md"
    path.write_text(kg_text, encoding="utf-8")

    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: offline_kg.OfflineKGService(path))
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)

    payload = pipeline._build_link_candidates("R_001", "进入帮助界面后点击帮助入口不显示")

    assert payload["pages"][0]["name"] == "帮助页面"
    assert payload["elements"][0]["name"] == "帮助按钮"
    assert payload["pages"][0]["evidence"]["report_hit"] is True
    assert payload["elements"][0]["evidence"]["report_hit"] is True


def test_local_element_extraction_uses_kg_operation_vocabulary(tmp_path, monkeypatch):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（1 个）
- 无响应

### Operation（操作行为）（1 个）
- 轻触

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（1 个）
- markers: 按钮、入口
- name: 帮助按钮 | surface: 帮助 | aliases: 帮助入口

### Page（页面实例）（1 个）
- markers: 页面、界面、页
- name: 帮助页面 | surface: 帮助

## 关系（Relations）

### OperatesOnElement（Operation → Element）
- 轻触 → 帮助按钮
""".strip()
    path = tmp_path / "KG_desc_op_vocab.md"
    path.write_text(kg_text, encoding="utf-8")

    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: offline_kg.OfflineKGService(path))
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)

    payload = pipeline._build_link_candidates("R_001", "轻触帮助入口后无响应")

    assert payload["elements"][0]["name"] == "帮助按钮"
    assert payload["elements"][0]["evidence"]["report_hit"] is True


def test_page_element_memory_only_used_as_fallback(monkeypatch):
    class _EmptyPageElementKG(_FakeKGService):
        def retrieve_link_candidates(self, query_text, top_k_op=4, top_k_history=0, top_k_instance=4):
            payload = super().retrieve_link_candidates(query_text, top_k_op=top_k_op, top_k_history=top_k_history, top_k_instance=top_k_instance)
            payload["pages"] = []
            payload["elements"] = []
            payload["page_confidence"] = 0.0
            payload["element_confidence"] = 0.0
            payload["page_evidence"] = {"direct_hit": False, "relation_hit": False, "lexicon_hit": False}
            payload["element_evidence"] = {"direct_hit": False, "relation_hit": False, "lexicon_hit": False}
            return payload

    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _EmptyPageElementKG())
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    pipeline._instance_store["Page"].append({"id": "Page:1", "name": "帮助页面", "embedding": [1.0]})
    pipeline._instance_store["Element"].append({"id": "Element:1", "name": "帮助按钮", "embedding": [1.0]})

    payload = pipeline._build_link_candidates("R_001", "点击帮助页面后帮助按钮无响应")

    assert payload["pages"]
    assert payload["elements"]
    assert payload["pages"][0]["source"] == "memory"
    assert payload["elements"][0]["source"] == "memory"
    assert payload["pages"][0]["evidence"]["memory_fallback"] is True


def test_tail_bias_prefers_later_page_match(tmp_path):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（1 个）
- 崩溃

### Operation（操作行为）（1 个）
- 进入

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（1 个）
- markers: 按钮
- name: 进入按钮 | surface: 进入

### Page（页面实例）（2 个）
- markers: 页面、界面、页
- name: 设置页面 | surface: 设置
- name: 帮助页面 | surface: 帮助

## 关系（Relations）
""".strip()
    path = tmp_path / "KG_desc_tail_page.md"
    path.write_text(kg_text, encoding="utf-8")

    service = offline_kg.OfflineKGService(path)
    candidates = service.retrieve_link_candidates("先进入设置页面 再进入帮助页面后崩溃", top_k_instance=2)

    assert candidates["pages"][0]["name"] == "帮助页面"
    assert candidates["pages"][0]["evidence"]["position_weight"] > candidates["pages"][1]["evidence"]["position_weight"]


def test_tail_bias_prefers_later_bug_match(tmp_path):
    kg_text = """
# 知识图谱描述

## 实体（Entities）

### BugSymptom（缺陷症状）（2 个）
- 未知错误
- 保存失败

### Operation（操作行为）（1 个）
- 保存

### Severity（风险等级）（3 个）
- ELEMENT
- PAGE
- PROGRAM

### Element（页面元素实例）（1 个）
- markers: 按钮
- name: 保存按钮 | surface: 保存

### Page（页面实例）（1 个）
- markers: 页面
- name: 保存页面 | surface: 保存

## 关系（Relations）
""".strip()
    path = tmp_path / "KG_desc_tail_bug.md"
    path.write_text(kg_text, encoding="utf-8")

    service = offline_kg.OfflineKGService(path)
    candidates = service.retrieve_link_candidates("点击保存前提示未知错误，最终修改保存失败", top_k_instance=2)

    assert candidates["bugs"][0]["name"] == "保存失败"
    assert candidates["bugs"][0]["evidence"]["position_weight"] > candidates["bugs"][1]["evidence"]["position_weight"]


def test_scope_post_calibration_prefers_local_rules(monkeypatch):
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)

    assert pipeline._post_calibrate_scope("点击后页面空白", "帮助页面", "未知组件", "program") == "page"
    assert pipeline._post_calibrate_scope("点击后帮助按钮无响应", "帮助页面", "帮助按钮", "program") == "element"
    assert pipeline._post_calibrate_scope("应用闪退并停止运行", "未知页面", "未知组件", "page") == "program"


def test_prompt_only_lists_unresolved_slots(monkeypatch):
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: _FakeKGService())
    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    payload = _make_manual_payload(
        report_id="R_001",
        text="点击帮助页面后内容异常",
        pages=[
            {"id": "P1", "name": "帮助页面", "score": 1.2, "source": "lexicon", "evidence": {"lexicon_hit": True}},
            {"id": "P2", "name": "设置页面", "score": 1.1, "source": "lexicon", "evidence": {"lexicon_hit": True}},
        ],
        elements=[],
        bugs=[{"id": "B1", "name": "内容异常", "score": 2.5, "source": "report", "evidence": {"report_hit": True}}],
        slot_prefill={
            "Operation": {"resolved": True, "value": "点击", "candidate_id": "O1", "reason": "test", "score": 2.0},
            "Page": {"resolved": False, "value": "", "candidate_id": "", "reason": "ambiguous", "score": 1.2},
            "Element": {"resolved": True, "value": "未知组件", "candidate_id": "E0", "reason": "no_candidates", "score": 0.0},
            "Bug": {"resolved": True, "value": "内容异常", "candidate_id": "B1", "reason": "test", "score": 2.5},
        },
        llm_slots=["Page"],
    )

    prompt = pipeline._build_candidate_report_block([payload])
    assert "Preset:" in prompt
    assert "Operation Candidates" not in prompt
    assert "Element Candidates" not in prompt
    assert "Bug Candidates" not in prompt
    assert "Page Candidates" in prompt
