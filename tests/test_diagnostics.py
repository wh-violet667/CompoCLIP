from collections import Counter
import re

import numpy as np

from src.diagnostics.perturbations import (
    PROBES,
    reverse_relation,
    shuffle_all_words,
    swap_subj_obj,
)


def words(text: str) -> Counter:
    return Counter(re.findall(r"[A-Za-z0-9']+", text.lower()))


def test_swap_preserves_bag():
    cap = "a dog chases a cat in the park"
    neg = swap_subj_obj(cap)
    assert neg is not None
    assert words(cap) == words(neg)
    assert neg != cap


def test_shuffle_preserves_bag():
    cap = "a man with glasses is wearing an orange hat"
    neg = shuffle_all_words(cap, None)
    assert neg is not None
    assert words(cap) == words(neg)


def test_reverse_relation():
    cap = "the door is to the left of the shirt"
    neg = reverse_relation(cap)
    assert neg == "the shirt to the left of the door"


def test_all_probes_are_bag_preserving():
    cap = "a red car is parked in front of a blue house"
    for fn in PROBES.values():
        neg = fn(cap)
        if neg is None:
            continue
        assert words(cap) == words(neg), (fn.__name__, neg)
