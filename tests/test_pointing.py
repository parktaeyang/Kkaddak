import numpy as np
import pytest

from kkaddak.fingers import FingerState, pointing_direction

W, H = 640, 480
POINTING = FingerState(extended=(False, True, False, False, False), bends=(0,) * 5)


def hand_pointing(angle_deg: float, length: float = 0.12) -> np.ndarray:
    """손바닥은 (0.5, 0.5) 주변 작은 덩어리, 검지 MCP→TIP은 angle 방향(화면 기준, 0=오른쪽, 위가 +)."""
    pts = np.zeros((21, 3))
    pts[:, :2] = np.random.default_rng(1).uniform(0.46, 0.54, (21, 2))
    mcp = np.array([0.5, 0.5])
    r = np.radians(angle_deg)
    # 정규화 좌표에서 픽셀 기준 각도가 되도록 x/y를 각각 화면 크기로 나눈다
    tip = mcp + np.array([np.cos(r) * length * H / W, -np.sin(r) * length])
    pts[5, :2], pts[8, :2] = mcp, tip
    pts[6, :2] = mcp + (tip - mcp) / 3
    pts[7, :2] = mcp + (tip - mcp) * 2 / 3
    return pts


@pytest.mark.parametrize("angle, expected", [(0, "right"), (180, "left"), (20, "right"), (-25, "right"), (160, "left")])
def test_horizontal_pointing(angle, expected):
    assert pointing_direction(POINTING, hand_pointing(angle), W, H) == expected


@pytest.mark.parametrize("angle", [90, -90, 60, 125])
def test_vertical_pointing_ignored(angle):
    assert pointing_direction(POINTING, hand_pointing(angle), W, H) is None


def test_other_fingers_extended_ignored():
    open_hand = FingerState(extended=(True,) * 5, bends=(0,) * 5)
    two = FingerState(extended=(False, True, True, False, False), bends=(0,) * 5)
    for state in (open_hand, two):
        assert pointing_direction(state, hand_pointing(0), W, H) is None


def test_thumb_does_not_matter():
    state = FingerState(extended=(True, True, False, False, False), bends=(0,) * 5)
    assert pointing_direction(state, hand_pointing(180), W, H) == "left"


def test_pointing_at_camera_ignored():
    # 카메라를 향해 가리키면 화면상 검지가 아주 짧게 보인다
    assert pointing_direction(POINTING, hand_pointing(0, length=0.01), W, H) is None
