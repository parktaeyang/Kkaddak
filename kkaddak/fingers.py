"""손가락 펴짐 판정.

MediaPipe의 world landmark(미터 단위, 손 중심 원점 3D 좌표)로 관절 굽힘 각도를 계산한다.
영상 좌표와 달리 x/y 축 척도가 같고 카메라와의 거리에 영향을 덜 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class FingerThresholds:
    # 검지~새끼: MCP/PIP/DIP 굽힘 각도 합(도)이 이 값 미만이면 펴짐
    finger_total_bend_max: float = 80.0
    # 엄지: MCP/IP 굽힘 합(도)이 이 값 미만이고
    thumb_total_bend_max: float = 60.0
    # 엄지 끝이 검지 MCP에서 (손바닥 크기 × 이 값) 이상 떨어져 있으면 펴짐
    thumb_spread_min: float = 0.6


@dataclass(frozen=True)
class FingerState:
    extended: tuple[bool, bool, bool, bool, bool]  # thumb, index, middle, ring, pinky
    bends: tuple[float, float, float, float, float]  # 디버그용 굽힘 합(도)

    @property
    def count(self) -> int:
        return sum(self.extended)

    @property
    def is_open_palm(self) -> bool:
        return all(self.extended)


def _bend_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """a→b 와 b→c 사이 각도. 일직선이면 0도, 완전히 접히면 180도."""
    v1 = b - a
    v2 = c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _palm_size(p: np.ndarray) -> float:
    # 손바닥이 돌아가도 한 길이에만 의존하지 않도록 여러 손바닥 길이의 평균을 쓴다
    lengths = [
        np.linalg.norm(p[INDEX[0]] - p[WRIST]),
        np.linalg.norm(p[MIDDLE[0]] - p[WRIST]),
        np.linalg.norm(p[PINKY[0]] - p[WRIST]),
        np.linalg.norm(p[PINKY[0]] - p[INDEX[0]]),
    ]
    return max(float(np.mean(lengths)), 1e-6)


def classify(points: Sequence[Sequence[float]], th: FingerThresholds = FingerThresholds()) -> FingerState:
    """21개 3D 랜드마크 → 손가락별 펴짐 여부."""
    p = np.asarray(points, dtype=float)
    if p.shape != (21, 3):
        raise ValueError(f"expected 21x3 landmarks, got {p.shape}")

    extended: list[bool] = []
    bends: list[float] = []

    cmc, mcp, ip, tip = (p[i] for i in THUMB)
    thumb_bend = _bend_deg(cmc, mcp, ip) + _bend_deg(mcp, ip, tip)
    thumb_spread = np.linalg.norm(tip - p[INDEX[0]]) / _palm_size(p)
    extended.append(thumb_bend < th.thumb_total_bend_max and thumb_spread > th.thumb_spread_min)
    bends.append(thumb_bend)

    for mcp_i, pip_i, dip_i, tip_i in (INDEX, MIDDLE, RING, PINKY):
        total = (
            _bend_deg(p[WRIST], p[mcp_i], p[pip_i])
            + _bend_deg(p[mcp_i], p[pip_i], p[dip_i])
            + _bend_deg(p[pip_i], p[dip_i], p[tip_i])
        )
        extended.append(total < th.finger_total_bend_max)
        bends.append(total)

    return FingerState(extended=tuple(extended), bends=tuple(bends))  # type: ignore[arg-type]


def is_upright(image_points: Sequence[Sequence[float]]) -> bool:
    """영상 좌표에서 손끝이 손목보다 위(손을 들어 올린 자세)인지. y는 아래로 증가."""
    p = np.asarray(image_points, dtype=float)
    return bool(p[MIDDLE[3], 1] < p[WRIST, 1] and p[MIDDLE[0], 1] < p[WRIST, 1])
