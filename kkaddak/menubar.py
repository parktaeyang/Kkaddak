"""메뉴바 앱.

메인 스레드는 AppKit(rumps)이 소유하고, 카메라·인식은 워커 스레드에서 돈다.
워커는 상태만 기록하고, 아이콘·메뉴 갱신은 메인 스레드 타이머가 담당한다.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time

import rumps
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSImage

from kkaddak import actions, config, keys
from kkaddak.engine import CameraError, Engine
from kkaddak.gesture import Phase

log = logging.getLogger(__name__)

ICON_ON = "hand.raised"
ICON_OFF = "hand.raised.slash"
ICON_ACTIVE = "hand.raised.fill"  # 손바닥 유지 중 / 방금 발동
FIRED_FLASH_S = 0.6
PERMISSION_CHECK_EVERY = 20  # 타이머 틱(0.1초) 단위
ACCESSIBILITY_SETTINGS = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
KEY_HELP = "예: s, shift+s, cmd+space, ctrl+up, f5, esc\n보조키: cmd, shift, ctrl, alt(option)"


def _symbol(name: str) -> NSImage:
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "kkaddak")
    img.setTemplate_(True)  # 다크/라이트 모드에 맞춰 색이 바뀜
    return img


class KkaddakApp(rumps.App):
    def __init__(self, engine: Engine):
        super().__init__("kkaddak", quit_button=None)
        self.engine = engine
        self._icons = {n: _symbol(n) for n in (ICON_ON, ICON_OFF, ICON_ACTIVE)}
        self._icon_name = ""
        self._set_icon(ICON_OFF)

        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._error: str | None = None
        self._phase = Phase.ARMED
        self._fired_at = 0.0
        self._recent: tuple[float, list[str]] = (0.0, [])
        self._ticks = 0
        self._blocked_seen = 0
        self._blocked_alert_shown = False

        self.status_item = rumps.MenuItem("꺼짐")
        self.recent_item = rumps.MenuItem("최근 동작 없음")
        self.gesture_items = {g: self._build_gesture_menu(g) for g in config.GESTURES}
        self.permission_item = rumps.MenuItem("키 입력 권한 확인 중…", callback=self.on_permission)
        self.toggle_item = rumps.MenuItem("켜기", callback=self.on_toggle)
        self.menu = [
            self.status_item,
            self.recent_item,
            None,
            rumps.MenuItem("제스처 설정"),
            *self.gesture_items.values(),
            None,
            self.permission_item,
            None,
            self.toggle_item,
            rumps.MenuItem("재시작", callback=self.on_restart),
            rumps.MenuItem("종료", callback=self.on_quit),
        ]
        self._refresh_gesture_menus()
        self._refresh_permission()
        if self._uses_keys() and not actions.has_key_permission():
            actions.request_key_permission()

        self._worker = threading.Thread(target=self._run_worker, name="engine", daemon=True)
        self._worker.start()
        self._enable(True)

    # ---- 메뉴 구성 (메인 스레드) ----

    def _build_gesture_menu(self, gesture: str) -> rumps.MenuItem:
        item = rumps.MenuItem(config.GESTURES[gesture])
        for action_type, label in actions.ACTION_TYPES.items():
            title = f"{label}…" if action_type == "key" else label
            item.add(rumps.MenuItem(title, callback=lambda _s, g=gesture, t=action_type: self.on_choose(g, t)))
        return item

    def _refresh_gesture_menus(self) -> None:
        for gesture, item in self.gesture_items.items():
            action = self.engine.bindings[gesture]
            item.title = f"    {config.GESTURES[gesture]}  →  {actions.describe(action)}"
            for sub, action_type in zip(item.values(), actions.ACTION_TYPES):
                sub.state = int(action["type"] == action_type)

    def _refresh_permission(self) -> None:
        if actions.has_key_permission():
            self.permission_item.title = "키 입력 권한: 허용됨"
            self.permission_item.set_callback(None)
        else:
            self.permission_item.title = "⚠︎ 키 입력 권한 허용하기… (허용 후 재시작)"
            self.permission_item.set_callback(self.on_permission)

    def _uses_keys(self) -> bool:
        return any(a["type"] == "key" for a in self.engine.bindings.values())

    def _set_binding(self, gesture: str, action: actions.Action) -> None:
        bindings = dict(self.engine.bindings)
        bindings[gesture] = action
        self.engine.set_bindings(bindings)
        try:
            config.save(bindings)
        except OSError as e:
            rumps.alert("설정을 저장하지 못했습니다", str(e))
        self._refresh_gesture_menus()
        if action["type"] == "key" and not actions.has_key_permission():
            self.on_permission(None)

    def _ask_key(self, gesture: str) -> str | None:
        current = self.engine.bindings[gesture]
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)  # 메뉴바 앱은 대화상자가 다른 창 뒤에 뜰 수 있음
        response = rumps.Window(
            message=KEY_HELP,
            title=f"{config.GESTURES[gesture]} — 입력할 키",
            default_text=current.get("key", ""),
            ok="저장",
            cancel="취소",
            dimensions=(260, 24),
        ).run()
        if not response.clicked:
            return None
        try:
            return keys.parse(response.text).spec
        except keys.KeySpecError as e:
            rumps.alert("키 형식이 올바르지 않습니다", f"{e}\n\n{KEY_HELP}")
            return None

    def _alert_blocked_keys(self) -> None:
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        clicked = rumps.alert(
            "키 입력이 차단되었습니다",
            "제스처는 인식했지만 손쉬운 사용 권한이 없어 키를 입력하지 못했습니다.\n\n"
            "시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용에서 Kkaddak을 켠 뒤, "
            "메뉴바 아이콘 > 재시작을 눌러 주세요.\n"
            "이미 켜져 있는데도 안 되면 목록에서 Kkaddak을 지우고(−) 다시 추가해 주세요.",
            ok="설정 열기",
            cancel="나중에",
        )
        if clicked == 1:
            actions.request_key_permission()
            subprocess.Popen(["open", ACCESSIBILITY_SETTINGS])

    # ---- 메뉴 콜백 (메인 스레드) ----

    def on_choose(self, gesture: str, action_type: str) -> None:
        if action_type == "key":
            spec = self._ask_key(gesture)
            if spec is None:
                return
            self._set_binding(gesture, {"type": "key", "key": spec})
        else:
            self._set_binding(gesture, {"type": action_type})

    def on_permission(self, _sender) -> None:
        if not actions.request_key_permission():
            subprocess.Popen(["open", ACCESSIBILITY_SETTINGS])
        self._refresh_permission()

    def on_toggle(self, _sender) -> None:
        self._enable(not self._enabled.is_set())

    def on_restart(self, _sender) -> None:
        # 손쉬운 사용 권한은 앱을 다시 시작해야 적용되는 경우가 많다
        bundle = _app_bundle_path()
        if bundle is None:
            rumps.alert("재시작", "앱(.app)으로 실행할 때만 재시작할 수 있습니다. 터미널에서 다시 실행해 주세요.")
            return
        # 현재 프로세스가 끝난 뒤 새로 띄우도록 잠시 기다렸다가 open 한다
        subprocess.Popen(["/bin/sh", "-c", f'sleep 1; open "{bundle}"'])
        self.on_quit(None)

    def on_quit(self, _sender) -> None:
        self._stop.set()
        self._enabled.set()  # 대기 중인 워커를 깨워 종료시킴
        self._worker.join(timeout=2)
        rumps.quit_application()

    # ---- 상태 갱신 (메인 스레드) ----

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
            self.status_item.title = "인식 중 · 제스처 감지"
        else:
            self._set_icon(ICON_ON)
            self.status_item.title = "인식 중"

        if actions.blocked_key_presses > self._blocked_seen and not self._blocked_alert_shown:
            self._blocked_seen = actions.blocked_key_presses
            self._blocked_alert_shown = True  # 세션당 한 번만 안내
            self._alert_blocked_keys()

        at, events = self._recent
        if events:
            names = ", ".join(config.GESTURES[g] for g in events)
            self.recent_item.title = f"최근: {names} ({time.strftime('%H:%M:%S', time.localtime(at))})"

        self._ticks += 1
        if self._ticks % PERMISSION_CHECK_EVERY == 0:
            self._refresh_permission()

    # ---- 워커 스레드 ----

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            self._enabled.wait()
            if self._stop.is_set():
                break
            try:
                self.engine.open()
                while self._enabled.is_set() and not self._stop.is_set():
                    result = self.engine.step()
                    if result.events:
                        self._fired_at = time.monotonic()
                        self._recent = (time.time(), result.events)
                    self._phase = self.engine.trigger.phase
            except CameraError as e:
                log.error("%s", e)
                self._error = str(e)
                self._enabled.clear()
            except Exception:
                log.exception("엔진 오류")
                self._error = "예상치 못한 오류가 발생했습니다. ~/Library/Logs/kkaddak/kkaddak.log를 확인하세요."
                self._enabled.clear()
            finally:
                self.engine.close()  # 끄면 카메라를 해제해 LED도 꺼짐
                self._phase = Phase.ARMED


def _app_bundle_path() -> str | None:
    """.app으로 실행 중이면 번들 경로 (…/Kkaddak.app/Contents/MacOS/Kkaddak 기준)."""
    if not getattr(sys, "frozen", False):
        return None
    exe = sys.executable
    marker = ".app/Contents/MacOS/"
    return exe[: exe.index(marker) + 4] if marker in exe else None


def run(engine: Engine) -> None:
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # Dock 아이콘 숨김
    KkaddakApp(engine).run()
