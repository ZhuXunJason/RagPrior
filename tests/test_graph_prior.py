import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph_prior


def _make_reports(total: int):
    return [{"description": f"报告 {idx} 点击按钮后无响应"} for idx in range(1, total + 1)]


def _extract_report_ids(prompt: str) -> list[str]:
    if "待处理报告：" in prompt:
        payload = prompt.split("待处理报告：", 1)[1]
    elif "输入报告：" in prompt:
        payload = prompt.split("输入报告：", 1)[1]
    else:
        payload = prompt

    report_ids = []
    for line in payload.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        report_id = line.split("\t", 1)[0].strip()
        if report_id.startswith("R_"):
            report_ids.append(report_id)
    return report_ids


def _build_item(report_id: str, same_fields: bool = False) -> dict[str, str]:
    suffix = "共享" if same_fields else report_id
    return {
        "id": report_id,
        "op": f"操作-{suffix}",
        "pg": f"页面-{suffix}",
        "el": f"组件-{suffix}",
        "bug": f"现象-{suffix}",
        "scope": "page" if report_id.endswith("1") else "element",
    }


def _fake_embeddings(texts, **_kwargs):
    return np.array([[float(len(text)), float(index + 1)] for index, text in enumerate(texts)], dtype=float)


def _fake_embeddings_unique(texts, **_kwargs):
    size = max(1, len(texts))
    return np.eye(size, dtype=float)


def _build_block(report_id: str, same_fields: bool = False) -> str:
    item = _build_item(report_id, same_fields=same_fields)
    return "\n".join(
        [
            f"ID: {item['id']}",
            f"OP: {item['op']}",
            f"PG: {item['pg']}",
            f"EL: {item['el']}",
            f"BUG: {item['bug']}",
            f"SCOPE: {item['scope']}",
            "###",
        ]
    )


def test_batch_size_groups_requests_and_limits_tokens(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        is_batch = "待处理报告：" in prompt
        call_log.append({"ids": report_ids, "max_token": max_token, "is_batch": is_batch})
        if is_batch:
            content = "\n".join(_build_block(report_id) for report_id in report_ids)
        else:
            content = _build_block(report_ids[0])
        return content, 10, 20

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=4, workers=1)
    sequence, score_map, _, _, _ = pipeline.run(_make_reports(9))

    batch_calls = [item for item in call_log if item["is_batch"]]
    single_calls = [item for item in call_log if not item["is_batch"]]

    assert len(batch_calls) == 3
    assert not single_calls
    assert [item["ids"] for item in batch_calls] == [
        ["R_001", "R_002", "R_003", "R_004"],
        ["R_005", "R_006", "R_007", "R_008"],
        ["R_009"],
    ]
    assert [item["max_token"] for item in batch_calls] == [1024, 1024, 256]
    assert len(sequence) == 9
    assert len(score_map) == 9


def test_short_key_batch_response_is_normalized(monkeypatch):
    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        content = "\n".join(
            [
                f"ID: {report_ids[0]}",
                "OP: 点击登录",
                "PG: 登录页",
                "EL: 登录按钮",
                "BUG: 闪退",
                "SCOPE: program",
                "###",
            ]
        )
        return content, 1, 1

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=1, workers=1)
    result = pipeline._extract_structured_batch([("R_001", "点击登录后闪退")], batch_index=1)

    assert result["R_001"] == {
        "Report_ID": "R_001",
        "Operation": "点击登录",
        "Page": "登录页",
        "Element": "登录按钮",
        "Bug": "闪退",
        "Severity_Scope": "program",
    }


def test_partial_batch_response_only_falls_back_for_missing_reports(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        is_batch = "待处理报告：" in prompt
        call_log.append({"ids": report_ids, "max_token": max_token, "is_batch": is_batch})
        if is_batch:
            content = _build_block("R_001")
        else:
            content = _build_block(report_ids[0])
        return content, 3, 4

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)
    sequence, _, _, _, _ = pipeline.run(_make_reports(2))

    batch_calls = [item for item in call_log if item["is_batch"]]
    single_calls = [item for item in call_log if not item["is_batch"]]

    assert len(batch_calls) == 1
    assert len(single_calls) == 1
    assert single_calls[0]["ids"] == ["R_002"]
    assert single_calls[0]["max_token"] == graph_prior.SINGLE_EXTRACTION_MAX_TOKENS
    assert sequence == [1, 2]


def test_missing_reports_are_retried_in_small_batches_before_single(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        is_batch = "待处理报告：" in prompt
        call_log.append({"ids": report_ids, "is_batch": is_batch})
        if is_batch and len(call_log) == 1:
            return _build_block("R_001"), 5, 0
        if is_batch:
            return "\n".join(_build_block(report_id) for report_id in report_ids), 5, 6
        return _build_block(report_ids[0]), 5, 6

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", _fake_embeddings_unique)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=4, workers=1)
    pipeline.run(_make_reports(4))

    assert call_log == [
        {"ids": ["R_001", "R_002", "R_003", "R_004"], "is_batch": True},
        {"ids": ["R_002", "R_003"], "is_batch": True},
        {"ids": ["R_004"], "is_batch": False},
    ]


def test_embedding_cache_reuses_identical_texts(monkeypatch):
    embedding_calls = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        content = "\n".join(_build_block(report_id, same_fields=True) for report_id in report_ids)
        return content, 2, 2

    def fake_embeddings(texts, **_kwargs):
        embedding_calls.append(list(texts))
        return _fake_embeddings(texts)

    monkeypatch.setattr(graph_prior, "ask_gpt", fake_ask_gpt)
    monkeypatch.setattr(graph_prior, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=2, workers=1)
    pipeline.run(_make_reports(3))

    assert len(embedding_calls) == 2
    assert embedding_calls[0] == ["报告 1 点击按钮后无响应", "报告 2 点击按钮后无响应", "报告 3 点击按钮后无响应"]
    assert embedding_calls[1] == ["操作-共享", "页面-共享", "组件-共享", "现象-共享"]


def test_near_duplicate_reports_only_send_representatives_to_llm(monkeypatch):
    call_log = []

    def fake_ask_gpt(prompt, model, max_token, base_url=None, api_key=None):
        report_ids = _extract_report_ids(prompt)
        call_log.append(report_ids)
        return "\n".join(_build_block(report_id) for report_id in report_ids), 3, 4

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
    monkeypatch.setattr(graph_prior, "get_kg_service", lambda **kwargs: None)

    reports = [
        {"description": "帮助页面一直加载不出内容"},
        {"description": "帮助页面一直加载不出内容 变体"},
        {"description": "点击收藏后列表不显示"},
        {"description": "点击收藏后列表不显示 变体"},
    ]

    pipeline = graph_prior.GraphPriorPipeline(model="fake-model", batch_size=4, workers=1)
    sequence, score_map, _, _, _ = pipeline.run(reports)

    assert call_log == [["R_001", "R_003"]]
    assert len(sequence) == 4
    assert len(score_map) == 4
