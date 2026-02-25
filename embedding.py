"""
Embedding-based Bug Report Deduplication and Prioritization.

Uses cosine similarity between report embeddings to detect duplicate bug reports.
Unique reports are prioritized first; duplicates are appended at the end.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from google import genai

from config import GOOGLE_API_KEY, OPENAI_API_KEY
from report_utils import compute_apsd_from_sequence

# Initialize Google client
google_client = genai.Client(api_key=GOOGLE_API_KEY)
batch = 100


def get_embeddings(texts: List[str], model: str = "gemini") -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        model: Model family name. If contains 'gemini', use Google; otherwise OpenAI.
    
    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    if "gemini" in model:
        return _get_gemini_embeddings(texts)
    else:
        return _get_openai_embeddings(texts)


def _get_gemini_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings using Google Gemini embedding model."""
    if not texts:
        return np.array([])

    all_vectors: List[List[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        result = google_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,  # type: ignore
        )
        embeddings = result.embeddings or []  # type: ignore
        all_vectors.extend([e.values for e in embeddings if e and e.values is not None])

    return np.array(all_vectors)


def _get_openai_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings using OpenAI embedding model."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,
    )
    # Sort by index to ensure correct ordering
    data = sorted(response.data, key=lambda x: x.index)
    return np.array([d.embedding for d in data])


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: np.ndarray of shape (n, d)
    
    Returns:
        np.ndarray of shape (n, n) with cosine similarities
    """
    # Normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms
    return normalized @ normalized.T


def deduplicate_reports(
    sim_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """
    Greedily deduplicate reports based on cosine similarity threshold.
    
    Iterates through reports in order. A report is marked as a duplicate
    if its similarity to any already-selected unique report >= threshold.
    
    Args:
        sim_matrix: Pairwise cosine similarity matrix (n x n).
        threshold: Similarity threshold. Reports with similarity >= threshold
                   to any already-selected report are considered duplicates.
    
    Returns:
        (unique_indices, duplicate_indices): 0-based indices.
         unique_indices preserves the original encounter order.
         duplicate_indices are the remaining reports.
    """
    n = sim_matrix.shape[0]
    unique_indices: List[int] = []
    duplicate_indices: List[int] = []
    
    for i in range(n):
        is_dup = False
        for u in unique_indices:
            if sim_matrix[i, u] >= threshold:
                is_dup = True
                break
        if is_dup:
            duplicate_indices.append(i)
        else:
            unique_indices.append(i)
    
    return unique_indices, duplicate_indices


def build_prioritized_sequence(
    unique_indices: List[int],
    duplicate_indices: List[int],
) -> List[int]:
    """
    Build the final prioritized sequence: unique reports first, then duplicates.
    Returns 1-based report numbers.
    """
    sequence = [i + 1 for i in unique_indices] + [i + 1 for i in duplicate_indices]
    return sequence


def run_embedding_mode(
    reports: List[Dict[str, Any]],
    thresholds: List[float],
    model: str = "gemini",
) -> Tuple[np.ndarray, Dict[float, List[int]]]:
    """
    Run embedding-based deduplication for multiple thresholds.
    
    Args:
        reports: List of report dicts with 'description' key.
        thresholds: List of similarity thresholds to evaluate.
        model: Model family for embedding computation.
    
    Returns:
        (sim_matrix, results_dict)
        - sim_matrix: The cosine similarity matrix.
        - results_dict: {threshold: prioritized_sequence (1-based report numbers)}
    """
    descriptions = [r["description"] for r in reports]
    
    print(f"Computing embeddings for {len(descriptions)} reports...")
    embeddings = get_embeddings(descriptions, model=model)
    print(f"Embedding shape: {embeddings.shape}")
    
    sim_matrix = cosine_similarity_matrix(embeddings)
    
    results: Dict[float, List[int]] = {}
    for t in thresholds:
        unique_idx, dup_idx = deduplicate_reports(sim_matrix, threshold=t)
        seq = build_prioritized_sequence(unique_idx, dup_idx)
        results[t] = seq
        print(f"  Threshold {t:.2f}: {len(unique_idx)} unique, {len(dup_idx)} duplicates")
    
    return sim_matrix, results


def compute_apsd(
    sequence: List[int],
    reports: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Compute APSD for a given prioritized sequence.
    
    APSD = 1 - sum(TF_i * S_i)/(n * sum(S_i)) + 1/(2n)
    where n = total reports, m = unique bug types.
    
    Args:
        sequence: 1-based report numbers in prioritized order.
        reports: List of report dicts with 'bug_id' key.
    
    Returns:
        APSD value, or None if not all bug types are covered.
    """
    return compute_apsd_from_sequence(sequence=sequence, reports=reports)
