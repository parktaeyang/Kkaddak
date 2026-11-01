"""자세 유지 트리거 상태머신.

시간은 모두 단조 시계 기준 밀리초. 카메라·추론과 분리된 순수 로직이라 단위 테스트가 가능하다.

  ARMED ──자세 감지──▶ HOLDING ──hold_ms 유지──▶ (발동) LATCHED
    ▲                    │ 자세가 dropout_ms 이상 끊김          │
    └────────────────────┘                                     │
    ▲            자세가 release_ms 이상 사라짐 + cooldown 경과   │
    └──────────────────────────────────────────────────────────┘

LATCHED 동안은 손바닥을 계속 펴고 있어도 재발동하지 않는다(재무장하려면 자세를 풀어야 함).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    ARMED = "armed"
    HOLDING = "holding"
    LATCHED = "latched"


@dataclass(frozen=True)
class HoldConfig:
    hold_ms: int = 500  # 이 시간 동안 자세를 유지해야 발동
    dropout_ms: int = 150  # HOLDING 중 이보다 짧은 인식 끊김은 무시
    release_ms: int = 400  # LATCHED 해제에 필요한 '자세 없음' 지속 시간
    cooldown_ms: int = 1500  # 발동 후 최소 재발동 간격


class HoldTrigger:
    def __init__(self, cfg: HoldConfig = HoldConfig()):
        self.cfg = cfg
        self.phase = Phase.ARMED
        self._hold_start: int | None = None
        self._last_seen: int | None = None
        self._fired_at: int | None = None

    def reset(self) -> None:
        """추적 상실·일시정지 등 외부 사유로 초기화. LATCHED는 유지해 즉시 재발동을 막는다."""
        if self.phase is Phase.HOLDING:
            self.phase = Phase.ARMED
        self._hold_start = None

    def progress(self, now_ms: int) -> float:
        """HOLDING 진행률 0~1 (UI 표시용)."""
        if self.phase is not Phase.HOLDING or self._hold_start is None:
            return 0.0
        return min(1.0, (now_ms - self._hold_start) / self.cfg.hold_ms)

    def update(self, now_ms: int, pose_active: bool) -> bool:
        """프레임마다 호출. 발동하는 프레임에서만 True."""
        cfg = self.cfg
        if pose_active:
            self._last_seen = now_ms

        if self.phase is Phase.ARMED:
            if pose_active:
                self.phase = Phase.HOLDING
                self._hold_start = now_ms
            return False

        if self.phase is Phase.HOLDING:
            assert self._hold_start is not None and self._last_seen is not None
            if not pose_active and now_ms - self._last_seen > cfg.dropout_ms:
                self.phase = Phase.ARMED
                self._hold_start = None
                return False
            if pose_active and now_ms - self._hold_start >= cfg.hold_ms:
                self.phase = Phase.LATCHED
                self._fired_at = now_ms
                return True
            return False

        # LATCHED
        assert self._fired_at is not None
        released = self._last_seen is None or now_ms - self._last_seen >= cfg.release_ms
        if released and now_ms - self._fired_at >= cfg.cooldown_ms:
            self.phase = Phase.ARMED
            self._hold_start = None
        return False


@dataclass(frozen=True)
class PushConfig:
    window_ms: int = 400  # 이 시간 안에
    grow_ratio: float = 1.3  # 손 크기가 이 배율 이상 커지면 '밀기'
    min_span_ms: int = 80  # 크기 비교 기준은 최소 이만큼 이전 프레임 (한 프레임 튐 무시)
    dropout_ms: int = 150  # 이보다 짧은 인식 끊김은 무시
    cooldown_ms: int = 700  # 발동 후 최소 재발동 간격
    retract_ratio: float = 0.85  # 발동 후 최대 크기의 이 비율 이하로 줄어야(손을 당겨야) 재무장


class PushDetector:
    """주먹을 카메라 쪽으로 미는 동작 감지.

    주먹이 유지되는 동안 손의 화면상 크기를 기록하고, window_ms 안에서 최소 크기 대비
    grow_ratio 이상 커지면 1회 발동한다. 이후 손을 당기거나 주먹을 풀어야 재무장된다.
    """

    def __init__(self, cfg: PushConfig = PushConfig()):
        self.cfg = cfg
        self._history: deque[tuple[int, float]] = deque()
        self._last_seen: int | None = None
        self._latched = False
        self._fired_at = 0
        self._peak = 0.0

    @property
    def latched(self) -> bool:
        return self._latched

    def reset(self) -> None:
        self._history.clear()
        self._last_seen = None

    def update(self, now_ms: int, fist_size: float | None) -> bool:
        """fist_size: 주먹일 때 손 크기(px), 주먹이 아니거나 손이 없으면 None. 발동 프레임에서만 True."""
        cfg = self.cfg
        if fist_size is None:
            if self._last_seen is None or now_ms - self._last_seen > cfg.dropout_ms:
                self._history.clear()
                if self._latched and now_ms - self._fired_at >= cfg.cooldown_ms:
                    self._latched = False
            return False

        self._last_seen = now_ms
        if self._latched:
            self._peak = max(self._peak, fist_size)
            if now_ms - self._fired_at >= cfg.cooldown_ms and fist_size <= self._peak * cfg.retract_ratio:
                self._latched = False
                self._history.clear()
            else:
                return False

        self._history.append((now_ms, fist_size))
        while self._history and now_ms - self._history[0][0] > cfg.window_ms:
            self._history.popleft()

        base = [s for t, s in self._history if now_ms - t >= cfg.min_span_ms]
        if base and fist_size / min(base) >= cfg.grow_ratio:
            self._latched = True
            self._fired_at = now_ms
            self._peak = fist_size
            self._history.clear()
            return True
        return False
