import pytest
from report_utils import validate_full_permutation, compute_apfd_from_sequence

def test_validate_full_permutation():
    assert validate_full_permutation([1, 2, 3, 4], 4) is True
    assert validate_full_permutation([1, 2, 3], 4) is False
    assert validate_full_permutation([1, 2, 4], 3) is False
    assert validate_full_permutation([1, 2, 2], 3) is False
    assert validate_full_permutation([], 0) is True

def test_compute_apfd_invalid_sequence():
    reports = [
        {"bug_type": "b1"},
        {"bug_type": "b2"},
        {"bug_type": "b3"}
    ]
    # Invalid sequence: length mismatch
    assert compute_apfd_from_sequence([1, 2], reports) is None
    # Invalid sequence: duplication
    assert compute_apfd_from_sequence([1, 2, 2], reports) is None
    # Invalid sequence: out of bounds
    assert compute_apfd_from_sequence([1, 2, 4], reports) is None

def test_compute_apfd_valid_sequence():
    reports = [
        {"bug_type": "b1"},
        {"bug_type": "b2"},
        {"bug_type": "b1"}
    ]
    # sequence: 1, 2, 3
    # bugs found at pos 1, 2. total pos_sum = 3. Note: bug_score map ensures we only care about first occurrences.
    score = compute_apfd_from_sequence([1, 2, 3], reports)
    assert score is not None
    assert 0 <= score <= 1.0
