"""DOCBOT-2911: value containment math."""

import pytest

from api.utils.value_containment import containment


def test_full_containment_of_smaller_set():
    assert containment(["a", "b"], ["a", "b", "c", "d"]) == 1.0


def test_partial_overlap_uses_min_size():
    assert containment(["a", "b", "c", "d"], ["c", "d", "e", "f", "g"]) == 0.5


def test_disjoint_is_zero():
    assert containment(["a"], ["b"]) == 0.0


def test_duplicates_ignored():
    assert containment(["a", "a", "b"], ["a"]) == 1.0


@pytest.mark.parametrize("a,b", [([], ["a"]), (["a"], []), (None, ["a"]), ([], [])])
def test_empty_side_is_none(a, b):
    assert containment(a, b) is None
