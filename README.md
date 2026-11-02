# kkaddak

맥북 카메라로 손 제스처를 인식해 macOS를 조작하는 메뉴바 앱.

## 제스처와 기본 동작

| 제스처 | 기본 동작 |
|---|---|
| 손바닥 들기 (다섯 손가락을 펴서 위로 들고 0.5초 유지) | `Space` 키 |
| 오른손 주먹 밀기 (주먹을 쥐고 카메라 쪽으로 빠르게 내밀기) | `S` 키 입력 |
| 왼손 주먹 밀기 | `A` 키 입력 |
| 왼쪽 가리키기 (검지만 펴서 가리키고 0.3초 유지) | `←` 방향키 |
| 오른쪽 가리키기 | `→` 방향키 |
| 위쪽 가리키기 | `↑` 방향키 |
| 아래쪽 가리키기 | `↓` 방향키 |
| 두 손가락 위로 (검지·중지만 펴서 위를 가리키고 0.3초 유지) | `Page Up` 키 |
| 두 손가락 아래로 | `Page Down` 키 |

손바닥 들기·가리키기·두 손가락에 키 입력을 지정하면 자세를 유지하는 동안 키를 누르고 있다가 손을 내리면 뗍니다. 게임처럼 매 프레임 키 상태를 확인하는 앱에서도 인식됩니다.

메뉴바 아이콘 > **제스처 설정**에서 제스처마다 동작을 바꿀 수 있습니다.
- `Mission Control` / `키 입력…` / `동작 없음`
- 키 입력 예: `s`, `shift+s`, `cmd+space`, `ctrl+up`, `f5`, `esc`
- 설정은 `~/Library/Application Support/kkaddak/config.json`에 저장됩니다.

키 입력은 실제 키보드의 키를 누르는 방식이라, 한글 입력 상태에서는 `s`가 `ㄴ`으로 입력됩니다.

## 앱 빌드 (독립 실행형)
```bash
./scripts/build_app.sh        # → dist/Kkaddak.app
```
`dist/Kkaddak.app`을 `/Applications`로 옮겨 더블클릭하면 실행됩니다. 처음 실행하면 다음 두 가지 권한을 허용해야 합니다.
- **카메라**: 실행 시 표시되는 허용 창
- **손쉬운 사용** (키 입력에 필요): 시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용에서 Kkaddak 허용

앱 아이콘(`assets/Kkaddak.icns`)은 `uv run --with pillow python scripts/make_icon.py`로 다시 만들 수 있습니다.

앱을 다시 빌드하면 서명이 바뀌어 권한을 다시 허용해야 할 수 있습니다. 로그는 `~/Library/Logs/kkaddak/kkaddak.log`에 남습니다.

> 현재는 개발용 ad-hoc 서명입니다. 다른 사람에게 배포하려면 Apple Developer ID로 서명하고 공증(notarization)을 받아야 Gatekeeper 경고 없이 실행됩니다. Apple Silicon(arm64) 전용입니다.

## 개발 실행
```bash
uv sync
curl -L --create-dirs -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
uv run python -m kkaddak                       # 메뉴바 앱
uv run python -m kkaddak --preview --dry-run   # 미리보기 창 + 로그만 (동작 실행 안 함)
uv run pytest                                  # 테스트
```
옵션: `--headless` (아이콘·창 없이 터미널에서만), `--hold-ms 500` (손바닥 유지 시간), `--no-upright` (손 세움 조건 끄기), `-v` (손가락별 굽힘 각도 로그)

미리보기 창에는 손마다 `Right`/`Left`, 펴진 손가락, 손 크기(px)가 표시됩니다. 오른손·왼손이 반대로 표시되면 알려주세요.

## 제스처·동작 추가 방법
- 제스처: `kkaddak/config.py`의 `GESTURES`·`DEFAULT_BINDINGS`에 등록 → `kkaddak/engine.py`에서 해당 id로 이벤트 발생
- 동작: `kkaddak/actions.py`의 `ACTION_TYPES`에 등록 → `perform()`/`describe()`에 분기 추가
