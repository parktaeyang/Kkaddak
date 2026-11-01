"""메뉴바 앱.

메인 스레드는 AppKit(rumps)이 소유하고, 카메라·인식은 워커 스레드에서 돈다.
워커는 상태만 기록하고, 아이콘 갱신은 메인 스레드 타이머가 담당한다.
"""

from __future__ import annotations

import logging
import threading
import time

import rumps
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSImage

from kkaddak.engine import CameraError, Engine, EngineConfig
from kkaddak.gesture import Phase

log = logging.getLogger(__name__)

ICON_ON = "hand.raised"
ICON_OFF = "hand.raised.slash"
ICON_ACTIVE = "hand.raised.fill"  # 손바닥 유지 중 / 방금 발동
FIRED_FLASH_S = 0.6


def _symbol(name: str) -> NSImage:
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "kkaddak")
    img.setTemplate_(True)  # 다크/라이트 모드에 맞춰 색이 바뀜
    return img


class KkaddakApp(rumps.App):
    def __init__(self, cfg: EngineConfig):
        super().__init__("kkaddak", quit_button=None)
        self.engine = Engine(cfg)
        self._icons = {n: _symbol(n) for n in (ICON_ON, ICON_OFF, ICON_ACTIVE)}
        self._icon_name = ""
        self._set_icon(ICON_OFF)

        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._error: str | None = None
        self._phase = Phase.ARMED
        self._fired_at = 0.0

        self.toggle_item = rumps.MenuItem("켜기", callback=self.on_toggle)
        self.status_item = rumps.MenuItem("꺼짐")
        self.menu = [self.status_item, None, self.toggle_item, None, rumps.MenuItem("종료", callback=self.on_quit)]

        self._worker = threading.Thread(target=self._run_worker, name="engine", daemon=True)
        self._worker.start()
        self._enable(True)

    # ---- 메인 스레드 ----

    def _set_icon(self, name: str) -> None:
        if name == self._icon_name:
            return
        self._icon_name = name
        # rumps의 icon 속성은 파일 경로만 받으므로 NSImage를 직접 지정
        self._icon_nsimage = self._icons[name]
        try:
            self._nsapp.setStatusBarIcon()
        except AttributeError:
            pass  # run() 전에는 상태바 항목이 아직 없음

    def _enable(self, on: bool) -> None:
        if on:
            self._error = None
            self._enabled.set()
        else:
            self._enabled.clear()
        self.toggle_item.title = "끄기" if on else "켜기"

    def on_toggle(self, _sender) -> None:
        self._enable(not self._enabled.is_set())

    def on_quit(self, _sender) -> None:
        self._stop.set()
        self._enabled.set()  # 대기 중인 워커를 깨워 종료시킴
        self._worker.join(timeout=2)
        rumps.quit_application()

    @rumps.timer(0.1)
    def _refresh(self, _timer) -> None:
        if self._error:
            # 번들이 아닌 스크립트 실행에서는 알림 센터를 쓸 수 없어 대화상자로 알림
            message, self._error = self._error, None
            self._enable(False)
            rumps.alert("kkaddak — 인식을 중지했습니다", message)

        if not self._enabled.is_set():
            self._set_icon(ICON_OFF)
            self.status_item.title = "꺼짐"
        elif time.monotonic() - self._fired_at < FIRED_FLASH_S or self._phase is Phase.HOLDING:
            self._set_icon(ICON_ACTIVE)
            self.status_item.title = "인식 중 · 손바닥 감지"
        else:
            self._set_icon(ICON_ON)
            self.status_item.title = "인식 중"

    # ---- 워커 스레드 ----

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            self._enabled.wait()
            if self._stop.is_set():
                break
            try:
                self.engine.open()
                while self._enabled.is_set() and not self._stop.is_set():
                    if self.engine.step().fired:
                        self._fired_at = time.monotonic()
                    self._phase = self.engine.trigger.phase
            except CameraError as e:
                log.error("%s", e)
                self._error = str(e)
                self._enabled.clear()
            except Exception:
                log.exception("엔진 오류")
                self._error = "예상치 못한 오류가 발생했습니다. 터미널 로그를 확인하세요."
                self._enabled.clear()
            finally:
                self.engine.close()  # 끄면 카메라를 해제해 LED도 꺼짐
                self._phase = Phase.ARMED


def run(cfg: EngineConfig) -> None:
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # Dock 아이콘 숨김
    KkaddakApp(cfg).run()
