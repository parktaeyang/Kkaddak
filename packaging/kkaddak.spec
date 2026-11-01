# PyInstaller 설정. 빌드: scripts/build_app.sh
# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
VERSION = "0.2.0"
BUNDLE_ID = "com.parktaeyang.kkaddak"

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(ROOT / "models" / "hand_landmarker.task"), "models"), *collect_data_files("mediapipe")],
    hiddenimports=["kkaddak.menubar"],
    # 손 인식에 필요 없는 무거운 패키지 (matplotlib은 mediapipe가 import 시점에 불러오므로 제외 불가)
    excludes=["tkinter", "sounddevice", "jax", "jaxlib", "tensorflow", "torch", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Kkaddak",
    console=False,
    target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, name="Kkaddak")
app = BUNDLE(
    coll,
    name="Kkaddak.app",
    bundle_identifier=BUNDLE_ID,
    version=VERSION,
    info_plist={
        "CFBundleName": "Kkaddak",
        "CFBundleDisplayName": "Kkaddak",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSUIElement": True,  # 메뉴바 전용 (Dock 아이콘 없음)
        "LSMinimumSystemVersion": "13.0",
        "NSCameraUsageDescription": "손 제스처를 인식하기 위해 카메라를 사용합니다. 영상은 저장하거나 전송하지 않습니다.",
        "NSHighResolutionCapable": True,
    },
)
