from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DedupResult:
    representatives: List[int]
    rep_to_members: Dict[int, List[int]]
    member_to_rep: Dict[int, int]
    threshold: float
    model: str
    total_reports: int


def _as_model_name(model: str) -> str:
    model = model.strip()
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _extract_embedding_values(item: Any) -> List[float]:
    if item is None:
        return []
    values = getattr(item, "values", None)
    if values is not None:
        return list(values)

    if isinstance(item, dict):
        if "values" in item:
            return list(item["values"])
        nested = item.get("embedding")
        if isinstance(nested, dict) and "values" in nested:
            return list(nested["values"])

    nested = getattr(item, "embedding", None)
    if nested is not None:
        nested_values = getattr(nested, "values", None)
        if nested_values is not None:
            return list(nested_values)

    return []


def _embed_texts_google(
    texts: List[str],
    api_key: str,
    model: str,
    batch_size: int = 32,
) -> np.ndarray:
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError(
            "google-genai is required for embedding pre-clustering. "
            "Please install dependencies from requirements.txt."
        ) from exc

    client = genai.Client(api_key=api_key)
    model_name = _as_model_name(model)

    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model_name,
            contents=batch,
        )

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            single = getattr(response, "embedding", None)
            embeddings = [single] if single is not None else []

        for item in embeddings:
            values = _extract_embedding_values(item)
            if not values:
                raise RuntimeError("Embedding API returned empty vector.")
            vectors.append(values)

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)}, expected {len(texts)}"
        )

    return np.asarray(vectors, dtype=np.float32)


def deduplicate_reports_by_embedding(
    reports: List[Dict[str, Any]],
    threshold: float,
    model: str,
    api_key: Optional[str],
) -> DedupResult:
    """
    Cluster reports by description embedding similarity.

    Returns an identity mapping when api_key is missing so pipeline still works.
    """
    total = len(reports)
    representatives: List[int] = []
    rep_to_members: Dict[int, List[int]] = {}
    member_to_rep: Dict[int, int] = {}

    if total == 0:
        return DedupResult([], {}, {}, threshold, model, 0)

    if not api_key:
        for idx in range(1, total + 1):
            representatives.append(idx)
            rep_to_members[idx] = []
            member_to_rep[idx] = idx
        return DedupResult(representatives, rep_to_members, member_to_rep, threshold, model, total)

    descriptions = [str(r.get("description", "") or "") for r in reports]
    vectors = _embed_texts_google(descriptions, api_key=api_key, model=model)

    # Normalize for cosine similarity via dot product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    assigned = set()

    for i in range(total):
        if i in assigned:
            continue

        rep_num = i + 1
        representatives.append(rep_num)
        rep_to_members[rep_num] = []
        member_to_rep[rep_num] = rep_num

        for j in range(i + 1, total):
            if j in assigned:
                continue
            score = float(np.dot(vectors[i], vectors[j]))
            if score >= threshold:
                member_num = j + 1
                rep_to_members[rep_num].append(member_num)
                member_to_rep[member_num] = rep_num
                assigned.add(j)

    return DedupResult(
        representatives=representatives,
        rep_to_members=rep_to_members,
        member_to_rep=member_to_rep,
        threshold=threshold,
        model=model,
        total_reports=total,
    )


def select_representative_reports(
    reports: List[Dict[str, Any]],
    dedup: DedupResult,
) -> List[Dict[str, Any]]:
    return [reports[num - 1] for num in dedup.representatives]


def expand_ranked_sequence(
    representative_sequence: List[int],
    dedup: DedupResult,
    total_reports: int,
) -> List[int]:
    """
    Expand ranked representatives back to full report sequence.

    APFD-oriented behavior:
    1) Keep representative order from upstream ranking.
    2) Build per-representative cluster queues: [rep, member1, member2, ...].
    3) Interleave clusters round-robin by position to maximize early diversity.
    4) Append missing report numbers at the tail for full coverage.
    """
    full: List[int] = []
    seen = set()

    clusters: List[List[int]] = []
    added_reps = set()

    for rep in representative_sequence:
        if rep < 1 or rep > total_reports:
            continue
        if rep in added_reps:
            continue

        queue = [rep]
        queue.extend(dedup.rep_to_members.get(rep, []))
        clusters.append(queue)
        added_reps.add(rep)

    max_len = max((len(c) for c in clusters), default=0)
    for i in range(max_len):
        for cluster in clusters:
            if i >= len(cluster):
                continue

            num = cluster[i]
            if num < 1 or num > total_reports:
                continue
            if num in seen:
                continue

            full.append(num)
            seen.add(num)

    for n in range(1, total_reports + 1):
        if n not in seen:
            full.append(n)
            seen.add(n)

    return full
