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
