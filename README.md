# kkaddak

맥북 카메라로 손 제스처를 인식해 macOS를 조작하는 가상 트랙패드.

## 현재 기능 (M1)
손바닥(다섯 손가락)을 펴서 들어 올리고 0.5초 유지 → **Mission Control** 실행

## 실행
```bash
uv sync
curl -L --create-dirs -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
uv run python -m kkaddak                       # 메뉴바 앱으로 실행
uv run python -m kkaddak --preview --dry-run   # 개발용: 미리보기 창 + 로그만 (동작 실행 안 함)
uv run pytest                                  # 테스트
```
옵션: `--headless` (아이콘·창 없이 터미널에서만), `--hold-ms 500` (유지 시간), `--no-upright` (손 세움 조건 끄기), `-v` (손가락별 굽힘 각도 로그)

기본 실행 시 메뉴바에 손 아이콘이 생깁니다.
- ✋ 인식 중 / 채워진 ✋ 손바닥 감지·발동 / 사선 그어진 손 꺼짐(카메라 해제)
- 아이콘 메뉴에서 켜기/끄기, 종료

처음 실행하면 터미널 앱에 카메라 권한을 요청합니다. `--preview`/`--headless` 모드는 미리보기 창에서 `q`/`ESC` 또는 `Ctrl+C`로 종료합니다.
