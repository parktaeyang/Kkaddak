"""macOS 동작 실행.

Mission Control은 키보드 단축키(Ctrl+↑) 대신 시스템 앱을 직접 실행한다.
사용자 단축키 설정에 의존하지 않고, 손쉬운 사용(Accessibility) 권한도 필요 없다.
"""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


def mission_control(dry_run: bool = False) -> None:
    if dry_run:
        log.info("[dry-run] Mission Control")
        return
    # Popen으로 띄워 카메라 루프를 막지 않는다
    subprocess.Popen(["open", "-a", "Mission Control"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("Mission Control 실행")
