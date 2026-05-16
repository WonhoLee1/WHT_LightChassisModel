# Control Center (Monitor UI) 기능 강화 및 버그 수정 계획

본 계획은 `monitor_ui.py`를 수정하여 사용자가 요청한 "창 가림 방지" 기능을 추가하고, 시뮬레이션 중단/리셋 시 발생하는 그래프 갱신 버그를 해결하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> - '항상 위' 기능을 위해 `Qt.WindowStaysOnTopHint` 플래그를 사용합니다. 이 기능을 켜면 다른 모든 창(무조코 포함)보다 앞에 위치하게 됩니다.
> - 리셋(Reset) 감지는 `iter=0` 데이터가 들어왔을 때 기존 히스토리가 있는 경우로 판단하여 자동으로 초기화하도록 구현할 예정입니다.

## Proposed Changes

### [wht_topo](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo)

#### [MODIFY] [monitor_ui.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/monitor_ui.py)

- **UI 구조 개선**:
    - 중앙 위젯 상단에 `QHBoxLayout`을 추가하여 전역 컨트롤(상태 레이블, 중단 버튼, '항상 위' 토글 버튼)을 배치합니다.
    - `self.stop_btn`, `self.status_label` 등 선언만 되어 있고 배치가 누락된 위젯들을 실제 화면에 표시합니다.
- **'항상 위' 기능 구현**:
    - `self.top_btn` (또는 `self.stay_on_top_btn`) 추가.
    - 토글 시 `setWindowFlags`를 호출하여 창 순서를 제어합니다.
    - 단축키 `Ctrl+T`를 연결합니다.
- **리셋 및 그래프 버그 수정**:
    - `update_data` 메서드에서 `data["iter"] == 0`이고 기존 데이터가 있는 경우 `self._clear_history()`를 호출하여 UI를 초기화합니다.
    - `_update_height_plot`에서 `tricontour` 오류 발생 시에도 최소한의 `scatter`는 유지되도록 보완하고, `canvas.draw()` 호출을 확실히 수행합니다.
    - 시뮬레이션 중단(STOP) 신호 수신 시에도 현재까지의 데이터는 마지막으로 한 번 더 갱신하도록 로직을 수정합니다.

## Verification Plan

### Automated Tests
- `python wht_topo/monitor_ui.py`를 직접 실행하여 (테스트 데이터 주입 로직 추가 필요) UI 레이아웃과 버튼 동작을 확인합니다.

### Manual Verification
- `run_topo.py`를 실행하여 모니터 창이 뜨는지 확인.
- '항상 위' 버튼을 눌러 MuJoCo 창 뒤로 숨지 않는지 확인.
- 최적화 도중 중단 후 슬라이더를 움직여 그래프가 바뀌는지 확인.
- 다시 실행(Reset 후 시작) 시 그래프가 처음부터 다시 그려지는지 확인.
