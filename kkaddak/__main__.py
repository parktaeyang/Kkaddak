"""손바닥(다섯 손가락)을 펴서 들어 올리면 Mission Control 실행.

실행: uv run python -m kkaddak             # 메뉴바 앱
      uv run python -m kkaddak --preview   # 개발용 미리보기 창
      uv run python -m kkaddak --headless  # 창·아이콘 없이 터미널에서만
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from kkaddak.engine import DEFAULT_MODEL, CameraError, Engine, EngineConfig, StepResult
from kkaddak.fingers import FINGER_NAMES
from kkaddak.gesture import HoldConfig, Phase

log = logging.getLogger("kkaddak")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="다섯 손가락 → Mission Control")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="메뉴바 대신 디버그 미리보기 창으로 실행")
    mode.add_argument("--headless", action="store_true", help="메뉴바·창 없이 터미널에서만 실행")
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="동작을 실행하지 않고 로그만 출력")
    ap.add_argument("--hold-ms", type=int, default=HoldConfig.hold_ms, help="발동까지 손바닥 유지 시간")
    ap.add_argument("--no-upright", action="store_true", help="손을 위로 세운 자세 조건 끄기")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def draw_overlay(frame, r: StepResult, engine: Engine, fps: float) -> None:
    h, w = frame.shape[:2]
    trigger = engine.trigger
    if r.image_pts is not None:
        for x, y, _ in r.image_pts:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
    if r.state is not None:
        flags = " ".join(f"{n[0].upper() if e else '.'}" for n, e in zip(FINGER_NAMES, r.state.extended))
        cv2.putText(frame, f"fingers {r.state.count}  [{flags}]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    color = {Phase.ARMED: (200, 200, 200), Phase.HOLDING: (0, 200, 255), Phase.LATCHED: (0, 255, 0)}[trigger.phase]
    cv2.putText(frame, f"{trigger.phase.value}  {fps:4.1f}fps", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if trigger.phase is Phase.HOLDING:
        cv2.rectangle(frame, (10, 70), (10 + int(200 * trigger.progress(r.now_ms)), 82), color, -1)


def run_cli(engine: Engine, preview: bool) -> None:
    try:
        engine.open()
    except CameraError as e:
        sys.exit(str(e))

    fps, fps_t0, fps_n = 0.0, time.monotonic(), 0
    log.info("준비 완료. 손바닥을 펴서 들어 올리면 Mission Control이 실행됩니다. (종료: Ctrl+C%s)",
             " 또는 미리보기 창에서 q" if preview else "")
    try:
        while True:
            r = engine.step()
            fps_n += 1
            if (t := time.monotonic()) - fps_t0 >= 1.0:
                fps, fps_t0, fps_n = fps_n / (t - fps_t0), t, 0

            if preview and r.frame is not None:
                draw_overlay(r.frame, r, engine, fps)
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = EngineConfig(
        camera=args.camera,
        model=args.model,
        hold_ms=args.hold_ms,
        require_upright=not args.no_upright,
        dry_run=args.dry_run,
    )
    if args.preview or args.headless:
        run_cli(Engine(cfg), preview=args.preview)
    else:
        from kkaddak import menubar  # AppKit은 메뉴바 모드에서만 불러옴

        log.info("메뉴바 앱으로 실행합니다. 종료는 메뉴바 아이콘 > 종료")
        menubar.run(cfg)


if __name__ == "__main__":
    main()
