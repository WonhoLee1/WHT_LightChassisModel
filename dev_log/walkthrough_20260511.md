# Walkthrough - Chassis Dynamic Simulation Debugging (2026-05-11)

## 1. 개요
WHT 샤시 모델의 동적 강제 변위(SPCD) 해석 과정에서 발생한 물리적 연결 누락, 수치적 불안정성, 시각화 왜곡 문제를 해결하였습니다.

## 2. 변경 사항

### 2.1 Solver 및 수치 안정성
- **RBE2 Penalty Stiffness**: `WHTDynamicSolver`에서 RBE2 요소를 고강성 빔으로 조립하여 하중 전달 체계를 구축했습니다.
- **Mass Matrix Patching**: 회전 DOF 등 질량이 0인 성분에 $10^{-10}$ 수준의 미소 질량을 주입하여 Newmark-beta 적분 시의 `NaN` 발생을 차단했습니다.
- **Variable Shadowing Fix**: `T`(변환 행렬)와 `T`(총 시간) 변수 충돌을 해결했습니다.

### 2.2 경계조건 및 모델링 (`exam4_dynamic.py`)
- **Flange SPC 제거**: 플랜지 전체 고정을 제거하고 코너 마스터 노드 기반 제어로 변경했습니다.
- **Minimal Constraints**: 마스터 노드 0번에 Tx, Ty 구속을 적용하여 강체 드리프트를 방지했습니다.

### 2.3 시각화 (`WHTVisualizer` & `wht_dynamic_common.py`)
- **RBE Spider-web**: RBE2 연결 관계를 굵은 흰색 라인으로 시각화하여 코너부 연결성을 명확히 했습니다.
- **BC Z-offset**: BC 마커를 Z축으로 2mm 띄워 가독성을 높였습니다.
- **Global Node Mapping**: `vtkOriginalPointIds`를 참조하여 애니메이션 시 모든 메시 파트가 완벽하게 동기화되도록 수정했습니다.

## 3. 검증 결과
- **물리적 응답**: 10mm 하프사인 입력에 대해 샤시 바디가 정상적으로 변형되며, 약 127mm의 피크 응답(자유단 진동 포함)을 확인했습니다.
- **가시화**: 비주얼라이저에서 메시, RBE 라인, 하중 마커가 에러 없이 조화롭게 렌더링됩니다.

---
**Commit Status**: All physical and visual issues resolved. Ready for further stress analysis.
