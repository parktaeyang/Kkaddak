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

from kkaddak import actions, keys
from kkaddak.config import DEFAULT_BINDINGS, GESTURES, Bindings
from kkaddak.fingers import (
    FingerState,
    bbox_iou,
    classify,
    hand_size_px,
    is_fist,
    is_upright,
    pointing_direction,
)
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
POINT_GESTURE = {"left": "point_left", "right": "point_right"}
# 가리키기는 손바닥보다 짧게 유지해도 발동하고, 빨리 재무장된다
POINT_HOLD = HoldConfig(hold_ms=300, dropout_ms=150, release_ms=250, cooldown_ms=500)
# 가리키기에 키 입력을 매핑하면 가리키는 동안 키를 누르고 있는다 (게임은 매 프레임 키 상태를 확인하므로).
# 이 시간 이상 가리키는 자세가 안 보이면 뗀다. 짧은 인식 끊김에 키가 떨어졌다 다시 눌리지 않게 한다.
POINT_RELEASE_MS = 150
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
    pointing: str | None  # "left" / "right" / None


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
        self.point = {direction: HoldTrigger(POINT_HOLD) for direction in POINT_GESTURE}
        self._point_seen: dict[str, int] = {}
        self._held: dict[str, tuple[actions.Action, keys.KeyCombo]] = {}  # 누르고 있는 키 (제스처 id → 동작, 키)
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
        self._reset_detectors()

    def _reset_detectors(self) -> None:
        self.release_keys()
        self.trigger.reset()
        for detector in self.push.values():
            detector.reset()
        for trigger in self.point.values():
            trigger.reset()

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
            self._reset_detectors()
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
            hands.append(
                HandObs(
                    top.category_name,
                    top.score,
                    image_pts,
                    state,
                    hand_size_px(image_pts, w, h),
                    pointing_direction(state, image_pts, w, h),
                )
            )
            log.debug("%s fingers=%s bends=%s", top.category_name, state.extended, [round(b) for b in state.bends])
        if len(hands) > 1:
            hands = dedupe_hands(hands)

        bindings = self.bindings
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

        events += self._update_pointing(now_ms, hands, bindings)
        self._dispatch(events, bindings)
        return StepResult(now_ms, frame, hands, events)

    def _update_pointing(self, now_ms: int, hands: list[HandObs], bindings: Bindings) -> list[str]:
        """가리키기 발동 제스처를 반환하고, 가리키기를 멈췄거나 설정이 바뀐 제스처의 키를 뗀다."""
        events: list[str] = []
        for direction, trigger in self.point.items():
            gesture = POINT_GESTURE[direction]
            active = any(hand.pointing == direction for hand in hands)
            if active:
                self._point_seen[gesture] = now_ms
            if trigger.update(now_ms, active):
                events.append(gesture)
            elif gesture in self._held and (
                now_ms - self._point_seen.get(gesture, now_ms) > POINT_RELEASE_MS
                or bindings.get(gesture) != self._held[gesture][0]
            ):
                self._release(gesture)
        return events

    def _dispatch(self, events: list[str], bindings: Bindings) -> None:
        for gesture in events:
            action = bindings.get(gesture, {"type": "none"})
            log.info("제스처 %s → %s", GESTURES[gesture], actions.describe(action))
            try:
                if gesture in POINT_GESTURE.values() and action["type"] == "key":
                    self._hold(gesture, action)
                else:
                    actions.perform(action, dry_run=self.cfg.dry_run)
            except Exception:
                log.exception("동작 실행 실패: %s", action)

    def _hold(self, gesture: str, action: actions.Action) -> None:
        self._release(gesture)
        combo = keys.parse(action["key"])
        if self.cfg.dry_run:
            log.info("[dry-run] 키 누름 %s", combo.label)
        elif not actions.key_down(combo):
            return
        self._held[gesture] = (action, combo)

    def _release(self, gesture: str) -> None:
        held = self._held.pop(gesture, None)
        if held is None:
            return
        if self.cfg.dry_run:
            log.info("[dry-run] 키 뗌 %s", held[1].label)
        else:
            actions.key_up(held[1])

    def release_keys(self) -> None:
        """누르고 있는 키를 모두 뗀다. 카메라를 끄거나 인식이 끊길 때 키가 눌린 채 남지 않게 한다."""
        for gesture in list(self._held):
            try:
                self._release(gesture)
            except Exception:
                log.exception("키 떼기 실패: %s", gesture)
