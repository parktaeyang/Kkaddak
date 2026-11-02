#!/usr/bin/env bash
# 배포용 dist/Kkaddak.dmg 생성 (열면 Kkaddak.app을 Applications로 드래그하는 창)
# 먼저 scripts/build_app.sh로 dist/Kkaddak.app을 빌드한다.
set -euo pipefail
cd "$(dirname "$0")/.."

APP=dist/Kkaddak.app
DMG=dist/Kkaddak.dmg
[[ -d $APP ]] || { echo "$APP 이 없습니다. scripts/build_app.sh를 먼저 실행하세요." >&2; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create -volname Kkaddak -srcfolder "$STAGE" -fs HFS+ -format UDZO -imagekey zlib-level=9 -ov "$DMG" >/dev/null
du -h "$DMG"
