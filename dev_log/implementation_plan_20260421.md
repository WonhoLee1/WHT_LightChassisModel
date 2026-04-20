# 솔버 정밀성 및 안정성 개선 계획 (Solver Enhancement Plan)

검증 보고서에서 식별된 QUAD4의 Spurious Mode 및 TRIA3의 Membrane Locking 이슈를 해결하여 해석 신뢰도를 높이기 위한 계획입니다.

## User Review Required

> [!IMPORTANT]
> **Drilling Penalty 강도 조절**: Penalty 값을 너무 높이면 수치적 불안정성(Ill-conditioning)이 발생할 수 있고, 너무 낮으면 가짜 모드가 나타납니다. 요소 크기에 비례하여 자동으로 조절되는 로직을 도입할 예정입니다.
> **계산 비용**: 요소 정식화가 복잡해짐에 따라 조립 시간이 약간 증가할 수 있습니다.

## Proposed Changes

### 1. QUAD4 Drilling DOF 안정화 개선
단순 Penalty 방식에서 벗어나, 요소의 굽힘 강성에 비례하는 동적 Penalty 설정을 도입하여 Spurious Mode를 물리적 주파수 대역 밖으로 밀어냅니다.

#### [MODIFY] [wht_quad4_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_quad4_element.py)
- `Ktt` 산출 시 요소의 평균 대각 강성(Diagonal Stiffness)을 참조하도록 수정.
- Drilling 강성 행렬의 가우스 적분 차수 재검토.

### 2. TRIA3 Membrane Locking 완화 (Bubble Function 도입)
삼각형 요소의 과강성 현상을 줄이기 위해 Membrane 파트에 Bubble Function을 추가하여 내부 변형율을 보정합니다.

#### [MODIFY] [wht_tria3_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_tria3_element.py)
- B-matrix에 Bubble mode 기여분 추가.
- 요소 내부에서 응력 보간(Stress Smoothing) 로직 강화.

### 3. 수치적 안정성 (AUTOSPC) 고도화
자유도가 부족한 노드나 특이 행렬(Singular Matrix)을 사전에 감지하고 부드럽게 패치하는 로직을 강화합니다.

#### [MODIFY] [wht_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_solver.py)
- `_assemble_K_scipy` 내의 `AUTOSPC` 로직을 개선하여, 구속되지 않은 회전 자유도에 대해 정밀한 Penalty 부여.

---

## Open Questions

- [ ] **JAX 호환성**: 모든 요소 개선 사항이 자동 미분(JIT) 가능하도록 `numpy` 대신 `jax.numpy` 전환을 병행할까요? (현재는 하이브리드 상태)
- [ ] **고차 요소(QUAD8)**: Penalty 방식 대신 8절점 요소를 도입하는 것이 장기적으로 유리할 수 있는데, 이에 대한 의견을 여쭈어봅니다.

## Verification Plan

### Automated Tests
- `SolverVerification/verification_runner.py`를 재실행하여 1차부터 **5차 모드**까지의 오차율을 분석.
- QUAD4의 경우 Drilling Mode가 고주파 대역으로 이동하여 물리적인 1~5차 모드가 정확히 탐지되는지 확인.
- 인장 패치 테스트(`test_membrane_uniaxial`)의 TRIA3 오차율이 15%에서 5% 이내로 감소하는지 검증.

### Manual Verification
- `test_jaxSSO/exam2_shell_jaxSSO.py`를 실행하여 고유진동수 결과 변화 관찰.
