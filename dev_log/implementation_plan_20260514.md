# 최적화 모니터 UI 탐색 기능 강화 계획

최적화 진행 상황을 시각화하는 `WHTMonitorWindow`의 'Height Distribution' 탭에 사용자가 편리하게 이터레이션 간을 이동할 수 있도록 슬라이더(Slider)와 이전/다음(Prev/Next) 버튼을 추가합니다.

## User Review Required

> [!NOTE]
> 슬라이더와 버튼은 기존의 드롭다운 메뉴와 동기화되어 작동하며, 최신 결과(Latest) 상태와 개별 이터레이션 상태를 모두 지원합니다.

## Proposed Changes

### [wht_topo](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo)

#### [MODIFY] [monitor_ui.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/monitor_ui.py)
- **UI 구성 수정 (`_init_ui`)**:
  - `QSlider` (수평 방향) 및 `QPushButton` (이전/다음) 위젯을 추가합니다.
  - 레이아웃을 조정하여 드롭다운 메뉴 옆에 버튼과 슬라이더를 배치합니다.
- **이벤트 연결**:
  - 슬라이더 값 변경 시 드롭다운 인덱스를 변경하도록 연결합니다.
  - 이전/다음 버튼 클릭 시 드롭다운 인덱스를 증감하도록 연결합니다.
- **데이터 갱신 로직 수정 (`update_data`)**:
  - 새로운 데이터 수신 시 슬라이더의 최대 범위를 갱신합니다.
  - 현재 선택된 상태가 "Latest"인 경우 슬라이더도 최신 위치로 자동 이동하게 합니다.
- **상태 관리**:
  - 드롭다운의 인덱스 순서(최신이 앞쪽)와 슬라이더의 직관적 순서(오른쪽이 최신) 사이의 매핑 로직을 구현합니다.

## Verification Plan

### Manual Verification
- `run_topo.py`를 실행하여 모니터 UI를 띄웁니다.
- 최적화가 진행됨에 따라 슬라이더의 범위가 늘어나는지 확인합니다.
- 슬라이더를 드래그하거나 이전/다음 버튼을 눌렀을 때 'Height Distribution' 그래프가 해당 이터레이션의 데이터로 즉시 갱신되는지 확인합니다.
- 드롭다운 메뉴를 변경했을 때 슬라이더의 위치가 함께 동기화되는지 확인합니다.
