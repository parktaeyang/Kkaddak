import numpy as np
import pytest

from kkaddak.fingers import bbox_iou, classify, is_upright
from tests.handgen import FIST, OPEN, make_hand


def test_open_palm():
    s = classify(OPEN)
    assert s.extended == (True,) * 5
    assert s.is_open_palm and s.count == 5


def test_fist():
    s = classify(FIST)
    assert s.count == 0
    assert not s.is_open_palm


def test_slightly_bent_fingers_still_open():
    # 자연스럽게 편 손은 관절이 조금씩 굽어 있다
    s = classify(make_hand(((10, 15, 10),) * 4, thumb_bends=(10, 10)))
    assert s.is_open_palm


def test_hooked_finger_is_not_extended():
    # MCP는 펴고 PIP/DIP만 굽힌 갈고리 모양
    s = classify(make_hand(((0, 0, 0), (0, 90, 60), (0, 0, 0), (0, 0, 0))))
    assert s.extended == (True, True, False, True, True)
    assert not s.is_open_palm


@pytest.mark.parametrize("n_down", [1, 2, 3])
def test_counts(n_down):
    bends = [(0, 0, 0)] * 4
    for i in range(n_down):
        bends[3 - i] = (80, 100, 60)
    s = classify(make_hand(tuple(bends)))
    assert s.count == 5 - n_down


def test_thumb_tucked():
    s = classify(make_hand(thumb_open=False, thumb_bends=(30, 40)))
    assert s.extended[0] is False
    assert s.count == 4


def test_scale_and_rotation_invariant():
    # 손 크기와 방향이 달라져도 판정이 같아야 한다
    theta = np.radians(35)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    for pts in (OPEN, FIST):
        assert classify(pts @ rot.T * 1.4).extended == classify(pts).extended


def test_bad_shape():
    with pytest.raises(ValueError):
        classify(np.zeros((20, 3)))


def test_upright():
    img = np.zeros((21, 3))
    img[0] = (0.5, 0.8, 0)  # 손목 아래
    img[9] = (0.5, 0.6, 0)
    img[12] = (0.5, 0.4, 0)
    assert is_upright(img)
    img[[9, 12], 1] = (0.9, 0.95)  # 손끝이 아래를 향함
    assert not is_upright(img)


def test_bbox_iou():
    a = np.zeros((21, 3))
    a[:, :2] = np.random.default_rng(0).uniform(0.2, 0.4, (21, 2))
    assert bbox_iou(a, a) == pytest.approx(1.0)
    b = a.copy()
    b[:, 0] += 0.5  # 완전히 떨어진 손
    assert bbox_iou(a, b) == 0.0
