# [2026-05-10] exam4_dynamic.py BC 변경 및 시각화 버그 수정

## 🎯 목표
- Flange의 모든 구속 조건 제거 및 코너 4곳을 주요 지지점/입력점으로 변경.
- WHT Inspector에서 Boundary Condition 및 Load 마커가 보이지 않는 문제 해결.

## 📝 변경 사항
1.  **시각화 라이브러리 수정:**
    - `wht_modeler/wht_mesh_model.py`: `to_wht_result_data` 시 모든 SPC 노드를 "SPC" 노드 셋으로 내보내도록 수정.
    - `wht_solver/wht_dynamic_common.py`: `DynamicResult.to_wht_result_data`가 하중 그룹을 받아 `Applied_Load` 필드를 생성하도록 수정.
2.  **Exam 4 스크립트 수정:**
    - Flange 고정 로직 제거.
    - 코너 4곳에 Tx, Ty 구속 추가 (수평 강체 거동 방지).
    - 1번 코너에 Tz 구속 추가 (수직 강체 거동 방지).
    - `to_wht_result_data` 호출 시 `load_groups`를 전달하여 하중 시각화 활성화.

## ✅ 작업 목록 (Task List)
- [x] 라이브러리 백업 생성
- [x] `wht_mesh_model.py` 수정 (SPC 노드셋 추가)
- [x] `wht_dynamic_common.py` 수정 (하중 이력 계산 추가)
- [x] `exam4_dynamic.py` 수정 (BC 교체 및 시각화 데이터 전달)
- [x] 결과 검증 (Visualizer 실행 확인)
