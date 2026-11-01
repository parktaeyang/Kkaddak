#!/usr/bin/env bash
# 독립 실행형 Kkaddak.app 빌드 → dist/Kkaddak.app
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=models/hand_landmarker.task
if [[ ! -f $MODEL ]]; then
  curl -L --create-dirs -o "$MODEL" \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi

uv sync
uv run pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/kkaddak.spec

# 기본은 ad-hoc 서명. 다시 빌드할 때마다 macOS가 다른 앱으로 보므로 카메라·손쉬운 사용 권한을 다시 허용해야 한다.
# 고정 인증서를 쓰려면 CODESIGN_IDENTITY="인증서 이름"으로 지정한다. 만료된 인증서로 서명하면
# macOS가 앱을 악성으로 판단해 삭제할 수 있으므로, 지정된 인증서의 유효기간을 먼저 확인한다.
# 다른 사람에게 배포하려면 Developer ID Application 인증서 서명 + 공증이 필요하다.
IDENTITY=${CODESIGN_IDENTITY:--}
if [[ $IDENTITY != - ]]; then
  PEM=$(mktemp)
  security find-certificate -c "$IDENTITY" -p > "$PEM"
  if ! security verify-cert -c "$PEM" -p codeSign -q >/dev/null 2>&1; then
    rm -f "$PEM"
    echo "인증서가 만료되었거나 유효하지 않습니다: $IDENTITY" >&2
    exit 1
  fi
  rm -f "$PEM"
fi
codesign --force --deep --sign "$IDENTITY" dist/Kkaddak.app
codesign --verify --deep --strict dist/Kkaddak.app
du -sh dist/Kkaddak.app
