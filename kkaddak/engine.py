"""카메라 → 손 인식 → 제스처 판정 → 매핑된 동작 실행 파이프라인.

UI와 무관하게 한 프레임씩 처리한다. CLI는 메인 스레드에서, 메뉴바 앱은 워커 스레드에서 step()을 돌린다.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from kkaddak import actions
from kkaddak.config import DEFAULT_BINDINGS, GESTURES, Bindings
from kkaddak.fingers import FingerState, bbox_iou, classify, hand_size_px, is_fist, is_upright
from kkaddak.gesture import HoldConfig, HoldTrigger, PushDetector


def _resource_root() -> Path:
    # PyInstaller로 묶인 앱에서는 리소스가 sys._MEIPASS 아래에 있다
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


DEFAULT_MODEL = _resource_root() / "models" / "hand_landmarker.task"
MAX_READ_FAILURES = 30
CAMERA_PROMPT_TIMEOUT_S = 120
DUPLICATE_IOU = 0.5  # 두 손의 영역이 이 이상 겹치면 같은 손을 중복 인식한 것으로 본다
HANDS = ("Right", "Left")  # MediaPipe handedness 라벨 (거울 모드 입력 기준 사용자의 실제 손)
PUSH_GESTURE = {"Right": "push_right", "Left": "push_left"}
log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


def ensure_camera_access() -> None:
    """카메라 권한을 확인하고, 아직 정하지 않았으면 허용 대화상자의 응답을 기다린다.

    권한이 정해지기 전에 카메라를 열면 즉시 실패하므로 첫 실행에서 반드시 거쳐야 한다.
    """
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    if status == AV.AVAuthorizationStatusNotDetermined:
        answered = threading.Event()
        granted: list[bool] = []

        def done(ok: bool) -> None:
            granted.append(bool(ok))
            answered.set()

        AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, done)
        if not answered.wait(CAMERA_PROMPT_TIMEOUT_S):
            raise CameraError("카메라 권한 요청에 응답이 없습니다. 다시 켜서 허용해 주세요.")
        if granted and granted[0]:
            return
        status = AV.AVAuthorizationStatusDenied
    if status in (AV.AVAuthorizationStatusDenied, AV.AVAuthorizationStatusRestricted):
        raise CameraError(
            "카메라 권한이 거부되어 있습니다. 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 "
            "Kkaddak(또는 실행한 터미널 앱)을 허용한 뒤 다시 켜 주세요."
        )


@dataclass(frozen=True)
class EngineConfig:
    camera: int = 0
    model: Path = DEFAULT_MODEL
    hold_ms: int = HoldConfig.hold_ms
    require_upright: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class HandObs:
    label: str  # "Right" / "Left"
    score: float  # 좌우 판정 확신도
    image_pts: list[tuple[float, float, float]]
    state: FingerState
    size_px: float


@dataclass(frozen=True)
class StepResult:
    now_ms: int
    frame: np.ndarray | None  # 거울 모드로 뒤집은 BGR 프레임
    hands: list[HandObs]
    events: list[str]  # 이번 프레임에 발동한 제스처 id


def dedupe_hands(hands: list[HandObs]) -> list[HandObs]:
    """MediaPipe가 손 하나를 오른손·왼손 두 개로 겹쳐 인식하는 경우, 확신도가 높은 쪽만 남긴다."""
    kept: list[HandObs] = []
    for hand in sorted(hands, key=lambda h: h.score, reverse=True):
        if all(bbox_iou(hand.image_pts, k.image_pts) < DUPLICATE_IOU for k in kept):
            kept.append(hand)
    return kept


class Engine:
    def __init__(self, cfg: EngineConfig = EngineConfig(), bindings: Bindings | None = None):
        self.cfg = cfg
        self.bindings: Bindings = dict(bindings or DEFAULT_BINDINGS)
        self.trigger = HoldTrigger(HoldConfig(hold_ms=cfg.hold_ms))
        self.push = {hand: PushDetector() for hand in HANDS}
        self._cap: cv2.VideoCapture | None = None
        self._landmarker: vision.HandLandmarker | None = None
        self._last_ts = -1
        self._fail_count = 0

    def set_bindings(self, bindings: Bindings) -> None:
        # 다른 스레드에서 호출될 수 있으므로 dict 참조를 통째로 교체한다
        self.bindings = dict(bindings)

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if not self.cfg.model.exists():
            raise CameraError(f"모델 파일이 없습니다: {self.cfg.model}")
        ensure_camera_access()
        cap = cv2.VideoCapture(self.cfg.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise CameraError(
                "카메라를 열 수 없습니다. 다른 앱이 카메라를 사용 중이거나 카메라가 연결되어 있지 않습니다."
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
            num_hands=2,
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
        for detector in self.push.values():
            detector.reset()

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
            for detector in self.push.values():
                detector.reset()
            return StepResult(now_ms, None, [], [])
        self._fail_count = 0

        frame = cv2.flip(frame, 1)  # 거울 모드
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now_ms)

        hands: list[HandObs] = []
        for lms, world, handedness in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            image_pts = [(lm.x, lm.y, lm.z) for lm in lms]
            state = classify([(lm.x, lm.y, lm.z) for lm in world])
            top = handedness[0]
            hands.append(HandObs(top.category_name, top.score, image_pts, state, hand_size_px(image_pts, w, h)))
            log.debug("%s fingers=%s bends=%s", top.category_name, state.extended, [round(b) for b in state.bends])
        if len(hands) > 1:
            hands = dedupe_hands(hands)

        events: list[str] = []
        palm = any(
            hand.state.is_open_palm and (not self.cfg.require_upright or is_upright(hand.image_pts)) for hand in hands
        )
        if self.trigger.update(now_ms, palm):
            events.append("palm")

        for label, detector in self.push.items():
            # 같은 라벨의 손이 둘 잡히면 큰 쪽(카메라에 가까운 손)을 쓴다
            fists = [hand.size_px for hand in hands if hand.label == label and is_fist(hand.state)]
            if detector.update(now_ms, max(fists) if fists else None):
                events.append(PUSH_GESTURE[label])

        bindings = self.bindings
        for gesture in events:
            action = bindings.get(gesture, {"type": "none"})
            log.info("제스처 %s → %s", GESTURES[gesture], actions.describe(action))
            try:
                actions.perform(action, dry_run=self.cfg.dry_run)
            except Exception:
                log.exception("동작 실행 실패: %s", action)
        return StepResult(now_ms, frame, hands, events)
