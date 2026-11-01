import numpy as np

from kkaddak.engine import HandObs, dedupe_hands
from kkaddak.fingers import classify
from tests.handgen import FIST


def hand(label, score, x0):
    pts = np.zeros((21, 3))
    pts[:, 0] = np.linspace(x0, x0 + 0.2, 21)
    pts[:, 1] = np.linspace(0.3, 0.6, 21)
    return HandObs(label, score, [tuple(p) for p in pts], classify(FIST), 100.0)


def test_duplicate_detection_keeps_confident_one():
    kept = dedupe_hands([hand("Left", 0.6, 0.30), hand("Right", 0.9, 0.31)])
    assert [h.label for h in kept] == ["Right"]


def test_two_separate_hands_kept():
    kept = dedupe_hands([hand("Left", 0.9, 0.1), hand("Right", 0.9, 0.6)])
    assert sorted(h.label for h in kept) == ["Left", "Right"]
