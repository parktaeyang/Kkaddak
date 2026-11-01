"""테스트용 합성 손 랜드마크 생성기 (world 좌표, 미터 단위)."""

from __future__ import annotations

import numpy as np

SEG = (0.04, 0.025, 0.022)  # MCP→PIP, PIP→DIP, DIP→TIP 길이
FINGER_BASE_X = (0.025, 0.0, -0.02, -0.038)  # 검지~새끼 MCP x 위치


def _rot_x(v: np.ndarray, deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


def _rot_z(v: np.ndarray, deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def make_hand(
    finger_bends: tuple[tuple[float, float, float], ...] = ((0, 0, 0),) * 4,
    thumb_bends: tuple[float, float] = (0, 0),
    thumb_open: bool = True,
) -> np.ndarray:
    """finger_bends: 검지~새끼 각각 (MCP, PIP, DIP) 굽힘(도). thumb_bends: (MCP, IP)."""
    p = np.zeros((21, 3))
    wrist = np.zeros(3)
    p[0] = wrist

    # 엄지: 손바닥 옆으로 벌어지거나(open) 손바닥 쪽으로 붙음
    cmc = np.array([0.03, -0.02, 0.0])
    direction = np.array([0.7, -0.7, 0.0]) if thumb_open else np.array([-0.3, -0.95, -0.2])
    direction /= np.linalg.norm(direction)
    p[1] = cmc
    pt = cmc + direction * 0.035
    p[2] = pt
    for j, (length, bend) in enumerate(zip((0.03, 0.025), thumb_bends)):
        direction = _rot_z(direction, bend)
        pt = pt + direction * length
        p[3 + j] = pt

    for f, (bx, bends) in enumerate(zip(FINGER_BASE_X, finger_bends)):
        base = 5 + 4 * f
        mcp = np.array([bx, -0.085, 0.0])
        p[base] = mcp
        direction = (mcp - wrist) / np.linalg.norm(mcp - wrist)
        pt = mcp
        for j, (length, bend) in enumerate(zip(SEG, bends)):
            direction = _rot_x(direction, -bend)  # 손바닥 쪽(-z)으로 굽힘
            pt = pt + direction * length
            p[base + 1 + j] = pt
    return p


OPEN = make_hand()
FIST = make_hand(((80, 100, 60),) * 4, thumb_bends=(40, 50), thumb_open=False)
