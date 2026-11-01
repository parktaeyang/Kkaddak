"""카메라 손 제스처 → 매핑된 동작 실행.

실행: uv run python -m kkaddak             # 메뉴바 앱
      uv run python -m kkaddak --preview   # 개발용 미리보기 창
      uv run python -m kkaddak --headless  # 창·아이콘 없이 터미널에서만
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from pathlib import Path

import cv2

from kkaddak import config
from kkaddak.engine import DEFAULT_MODEL, CameraError, Engine, EngineConfig, StepResult
from kkaddak.fingers import FINGER_NAMES
from kkaddak.gesture import HoldConfig, Phase

LOG_PATH = Path.home() / "Library" / "Logs" / "kkaddak" / "kkaddak.log"
EVENT_SHOW_S = 1.0
log = logging.getLogger("kkaddak")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="카메라 손 제스처로 macOS 조작")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="메뉴바 대신 디버그 미리보기 창으로 실행")
    mode.add_argument("--headless", action="store_true", help="메뉴바·창 없이 터미널에서만 실행")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="동작을 실행하지 않고 로그만 출력")
    ap.add_argument("--hold-ms", type=int, default=HoldConfig.hold_ms, help="발동까지 손바닥 유지 시간")
    ap.add_argument("--no-upright", action="store_true", help="손을 위로 세운 자세 조건 끄기")
    ap.add_argument("-v", "--verbose", action="store_true")
    # macOS가 앱 실행 시 붙이는 -psn_ 인자 등 알 수 없는 인자는 무시
    args, _ = ap.parse_known_args()
    return args


def setup_logging(verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def draw_overlay(frame, r: StepResult, engine: Engine, fps: float, recent: list[str]) -> None:
    h, w = frame.shape[:2]
    trigger = engine.trigger
    for i, hand in enumerate(r.hands):
        color = (0, 255, 0) if hand.label == "Right" else (255, 160, 0)
        for x, y, _ in hand.image_pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
        flags = " ".join(f"{n[0].upper() if e else '.'}" for n, e in zip(FINGER_NAMES, hand.state.extended))
        text = f"{hand.label:<5} {hand.state.count} [{flags}] {hand.size_px:3.0f}px"
        cv2.putText(frame, text, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    color = {Phase.ARMED: (200, 200, 200), Phase.HOLDING: (0, 200, 255), Phase.LATCHED: (0, 255, 0)}[trigger.phase]
    cv2.putText(frame, f"palm {trigger.phase.value}  {fps:4.1f}fps", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if trigger.phase is Phase.HOLDING:
        cv2.rectangle(frame, (10, h - 50), (10 + int(200 * trigger.progress(r.now_ms)), h - 40), color, -1)
    if recent:
        # OpenCV 기본 글꼴은 한글을 그리지 못해 제스처 id로 표시
        cv2.putText(frame, " ".join(recent), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def run_cli(engine: Engine, preview: bool) -> None:
    try:
        engine.open()
    except CameraError as e:
        sys.exit(str(e))

    fps, fps_t0, fps_n = 0.0, time.monotonic(), 0
    recent: list[str] = []
    recent_at = 0.0
    log.info("준비 완료. (종료: Ctrl+C%s)", " 또는 미리보기 창에서 q" if preview else "")
    try:
        while True:
            r = engine.step()
            fps_n += 1
            if (t := time.monotonic()) - fps_t0 >= 1.0:
                fps, fps_t0, fps_n = fps_n / (t - fps_t0), t, 0

            if preview and r.frame is not None:
                if r.events:
                    recent, recent_at = r.events, time.monotonic()
                elif time.monotonic() - recent_at > EVENT_SHOW_S:
                    recent = []
                draw_overlay(r.frame, r, engine, fps, recent)
                cv2.imshow("kkaddak", r.frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except CameraError as e:
        log.error("%s", e)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        if preview:
            cv2.destroyAllWindows()
        log.info("종료")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    cfg = EngineConfig(
        camera=args.camera,
        model=args.model,
        hold_ms=args.hold_ms,
        require_upright=not args.no_upright,
        dry_run=args.dry_run,
    )
    engine = Engine(cfg, config.load())
    if args.preview or args.headless:
        run_cli(engine, preview=args.preview)
    else:
        from kkaddak import menubar  # AppKit은 메뉴바 모드에서만 불러옴

        log.info("메뉴바 앱으로 실행합니다. 종료는 메뉴바 아이콘 > 종료")
        menubar.run(engine)


if __name__ == "__main__":
    main()
