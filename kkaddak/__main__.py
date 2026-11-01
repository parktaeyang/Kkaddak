"""손바닥(다섯 손가락)을 펴서 들어 올리면 Mission Control 실행.

실행: uv run python -m kkaddak [--preview] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from kkaddak import actions
from kkaddak.fingers import FINGER_NAMES, classify, is_upright
from kkaddak.gesture import HoldConfig, HoldTrigger, Phase

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
log = logging.getLogger("kkaddak")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="다섯 손가락 → Mission Control")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--preview", action="store_true", help="디버그 미리보기 창 표시")
    ap.add_argument("--dry-run", action="store_true", help="동작을 실행하지 않고 로그만 출력")
    ap.add_argument("--hold-ms", type=int, default=HoldConfig.hold_ms, help="발동까지 손바닥 유지 시간")
    ap.add_argument("--no-upright", action="store_true", help="손을 위로 세운 자세 조건 끄기")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit(
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
    return cap


def create_landmarker(model: Path) -> vision.HandLandmarker:
    if not model.exists():
        sys.exit(f"모델 파일이 없습니다: {model}")
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def draw_overlay(frame, image_pts, state, trigger: HoldTrigger, now_ms: int, fps: float) -> None:
    h, w = frame.shape[:2]
    if image_pts is not None:
        for x, y, _ in image_pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
    if state is not None:
        flags = " ".join(f"{n[0].upper() if e else '.'}" for n, e in zip(FINGER_NAMES, state.extended))
        cv2.putText(frame, f"fingers {state.count}  [{flags}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    color = {Phase.ARMED: (200, 200, 200), Phase.HOLDING: (0, 200, 255), Phase.LATCHED: (0, 255, 0)}[trigger.phase]
    cv2.putText(frame, f"{trigger.phase.value}  {fps:4.1f}fps", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if trigger.phase is Phase.HOLDING:
        cv2.rectangle(frame, (10, 70), (10 + int(200 * trigger.progress(now_ms)), 82), color, -1)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cap = open_camera(args.camera)
    landmarker = create_landmarker(args.model)
    trigger = HoldTrigger(HoldConfig(hold_ms=args.hold_ms))
    last_ts = -1
    fps, fps_t0, fps_n = 0.0, time.monotonic(), 0
    fail_count = 0
    log.info("준비 완료. 손바닥을 펴서 들어 올리면 Mission Control이 실행됩니다. (종료: Ctrl+C%s)",
             " 또는 미리보기 창에서 q" if args.preview else "")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count > 30:
                    sys.exit("카메라 프레임을 읽지 못했습니다. 다른 앱이 카메라를 사용 중인지 확인하세요.")
                trigger.reset()
                continue
            fail_count = 0

            frame = cv2.flip(frame, 1)  # 거울 모드
            # VIDEO 모드는 엄격히 증가하는 타임스탬프가 필요
            now_ms = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = now_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), now_ms)

            image_pts = state = None
            pose_active = False
            if result.hand_landmarks:
                image_pts = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]
                world_pts = [(lm.x, lm.y, lm.z) for lm in result.hand_world_landmarks[0]]
                state = classify(world_pts)
                pose_active = state.is_open_palm and (args.no_upright or is_upright(image_pts))
                log.debug("fingers=%s bends=%s", state.extended, [round(b) for b in state.bends])

            if trigger.update(now_ms, pose_active):
                actions.mission_control(dry_run=args.dry_run)

            fps_n += 1
            if (t := time.monotonic()) - fps_t0 >= 1.0:
                fps, fps_t0, fps_n = fps_n / (t - fps_t0), t, 0

            if args.preview:
                draw_overlay(frame, image_pts, state, trigger, now_ms, fps)
                cv2.imshow("kkaddak", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()
        if args.preview:
            cv2.destroyAllWindows()
        log.info("종료")


if __name__ == "__main__":
    main()
