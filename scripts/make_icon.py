"""앱 아이콘(도트 손바닥) 생성.

실행: uv run --with pillow python scripts/make_icon.py
결과: assets/icon_1024.png, assets/Kkaddak.icns (iconutil 필요, macOS 기본 포함)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# 손바닥 모양 (#: 피부). 외곽선과 음영은 자동으로 붙인다.
HAND = """
.......##.......
....##.##.##....
....##.##.##....
....##.##.##.##.
....##.##.##.##.
....##.##.##.##.
....##.##.##.##.
....##.##.##.##.
##..###########.
###.###########.
.##############.
..#############.
...############.
....##########..
.....########...
.....########...
""".strip().splitlines()

SIZE = 1024
TILE = 824  # macOS 아이콘 격자: 1024 캔버스 안의 둥근 사각형
RADIUS = 185
BG_TOP, BG_BOTTOM = (64, 76, 214), (36, 34, 120)
SKIN, SHADE, LIGHT, OUTLINE = (255, 206, 84), (232, 160, 52), (255, 236, 160), (40, 24, 40)


def pixel_map() -> list[list[tuple[int, int, int] | None]]:
    """외곽선 1칸을 두른 색상 격자."""
    h, w = len(HAND) + 2, len(HAND[0]) + 2
    filled = {(y + 1, x + 1) for y, row in enumerate(HAND) for x, c in enumerate(row) if c == "#"}
    grid: list[list[tuple[int, int, int] | None]] = [[None] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if (y, x) in filled:
                if (y, x + 1) not in filled:
                    grid[y][x] = SHADE  # 오른쪽 가장자리 그림자
                elif (y, x - 1) not in filled:
                    grid[y][x] = LIGHT  # 왼쪽 가장자리 하이라이트
                else:
                    grid[y][x] = SKIN
            elif any((y + dy, x + dx) in filled for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                grid[y][x] = OUTLINE
    return grid


def render() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # 세로 그라데이션 배경을 둥근 사각형으로 잘라 붙인다
    grad = Image.new("RGBA", (TILE, TILE))
    gd = ImageDraw.Draw(grad)
    for y in range(TILE):
        t = y / (TILE - 1)
        gd.line([(0, y), (TILE, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)))
    mask = Image.new("L", (TILE, TILE), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, TILE - 1, TILE - 1], RADIUS, fill=255)
    off = (SIZE - TILE) // 2
    img.paste(grad, (off, off), mask)

    grid = pixel_map()
    cell = 34
    gw, gh = len(grid[0]) * cell, len(grid) * cell
    x0, y0 = (SIZE - gw) // 2, (SIZE - gh) // 2 + 10
    d = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color:
                d.rectangle([x0 + x * cell, y0 + y * cell, x0 + (x + 1) * cell - 1, y0 + (y + 1) * cell - 1], fill=color)
    return img


def make_icns(img: Image.Image, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                img.resize((px, px), Image.LANCZOS).save(iconset / name)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    img = render()
    img.save(ASSETS / "icon_1024.png")
    if shutil.which("iconutil"):
        make_icns(img, ASSETS / "Kkaddak.icns")
    print(f"생성: {ASSETS}")


if __name__ == "__main__":
    main()
