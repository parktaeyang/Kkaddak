"""카메라 → 손 인식 → 판정 → 동작 파이프라인.

UI와 무관하게 한 프레임씩 처리한다. CLI는 메인 스레드에서, 메뉴바 앱은 워커 스레드에서 step()을 돌린다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from kkaddak import actions
from kkaddak.fingers import FingerState, classify, is_upright
from kkaddak.gesture import HoldConfig, HoldTrigger

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MAX_READ_FAILURES = 30
log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineConfig:
    camera: int = 0
    model: Path = DEFAULT_MODEL
    hold_ms: int = HoldConfig.hold_ms
    require_upright: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class StepResult:
    now_ms: int
    frame: np.ndarray | None  # 거울 모드로 뒤집은 BGR 프레임
    image_pts: list[tuple[float, float, float]] | None
    state: FingerState | None
    fired: bool


class Engine:
    def __init__(self, cfg: EngineConfig = EngineConfig()):
        self.cfg = cfg
        self.trigger = HoldTrigger(HoldConfig(hold_ms=cfg.hold_ms))
        self._cap: cv2.VideoCapture | None = None
        self._landmarker: vision.HandLandmarker | None = None
        self._last_ts = -1
        self._fail_count = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if not self.cfg.model.exists():
            raise CameraError(f"모델 파일이 없습니다: {self.cfg.model}")
        cap = cv2.VideoCapture(self.cfg.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise CameraError(
                "카메라를 열 수 없습니다. 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 "
                "이 프로그램을 실행한 터미널 앱의 접근을 허용했는지 확인하세요."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        log.info(
            "카메라 %dx%d @%.0f",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FPS),
        )
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.cfg.model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._cap = cap
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._fail_count = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self.trigger.reset()

    def step(self) -> StepResult:
        assert self._cap is not None and self._landmarker is not None, "open() 먼저 호출"
        ok, frame = self._cap.read()
        # VIDEO 모드는 엄격히 증가하는 타임스탬프가 필요
        now_ms = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = now_ms

        if not ok:
            self._fail_count += 1
            if self._fail_count > MAX_READ_FAILURES:
                raise CameraError("카메라 프레임을 읽지 못했습니다. 다른 앱이 카메라를 사용 중인지 확인하세요.")
            self.trigger.reset()
            return StepResult(now_ms, None, None, None, False)
        self._fail_count = 0

        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now_ms)

        image_pts = state = None
        pose_active = False
        if result.hand_landmarks:
            image_pts = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
            world_pts = [(lm.x, lm.y, lm.z) for lm in result.hand_world_landmarks[0]]
            state = classify(world_pts)
            pose_active = state.is_open_palm and (not self.cfg.require_upright or is_upright(image_pts))
            log.debug("fingers=%s bends=%s", state.extended, [round(b) for b in state.bends])

        fired = self.trigger.update(now_ms, pose_active)
        if fired:
            actions.mission_control(dry_run=self.cfg.dry_run)
        return StepResult(now_ms, frame, image_pts, state, fired)
