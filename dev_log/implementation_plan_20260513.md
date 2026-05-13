# Nodal Stress & Strain Recovery Implementation Plan

## 문제 원인 분석
현재 `ElementStressRecovery`는 QUAD4/TRIA3 요소의 중심(Centroid, $\xi=0, \eta=0$)에서 단일 1점(1-point) 적분으로 응력과 변형률을 계산합니다. 판재의 동적 벤딩 시, 경계조건(SPCD)이 인가되는 코너 노드 부근에서 곡률(Curvature)이 급격히 증가하는 국부적인 피크(Peak)가 발생합니다. 그러나 요소 중심에서 평균화된 곡률만 계산하므로, 굽힘 응력이 현저히 낮게 평가되고 로컬한 응력 집중이 시각적으로 나타나지 않는 현상이 발생합니다.

## User Review Required
> [!IMPORTANT]
> Nodal Stress Averaging(절점 응력 평균화) 기법을 도입하여, 각 요소의 절점(Node) 위치($\xi=\pm 1, \eta=\pm 1$)에서 형상함수 미분값을 평가한 후, 이를 공유하는 노드별로 평균 내어 **PointData**로 출력하도록 구조를 변경하고자 합니다. 
> 이 방식은 ParaView에서 `CellData` 대신 `PointData`로 부드럽고 정확한 응력 분포(로컬 피크 포함)를 보여주게 됩니다. 진행해도 좋을지 검토 부탁드립니다.

## Proposed Changes

### 1. `wht_solver/wht_stress_recovery.py`
Nodal Stress 평가를 위해 기존 1점(Centroid) 평가 대신, 4개 노드 위치에서 Jacobian 및 변형률을 계산하는 모드를 추가합니다.
#### [MODIFY] wht_stress_recovery.py
- `recover_quad4_nodal()` 및 `recover_tria3_nodal()` 메서드 신설.
- $\xi, \eta = (\pm 1, \pm 1)$ 위치에서 `dNxi`, `dNeta`를 4번(각 노드별로) 평가하여 각 노드에서의 `kappa` 및 `eps_m` 계산.
- 각 요소의 4개 절점에서의 로컬 응력을 전역 Voigt로 변환하여 `(M_total, 4, 6)` 형태로 반환.

### 2. `wht_solver/wht_dynamic_solver.py`
`recover_stress_history` 메서드를 수정하여, Element Nodal Stress를 Global Nodal Stress로 평균화(Averaging)하는 로직을 추가합니다.
#### [MODIFY] wht_dynamic_solver.py
- `recover_stress_history`에서 `ElementStressRecovery.recover_quad4_nodal` 호출.
- 얻어진 `(M, 4, 6)` 요소 절점 응력을 전체 절점 `(N_nodes, 6)` 배열에 누적하고, 노드별 연결된 요소 개수로 나누어 평균화.
- 생성된 Nodal Stress 배열을 `dynamic_result.stress_data`에 저장.
- `DynamicResult.to_wht_result_data` 및 `_build_cell_data` 관련 로직을 수정하여 `stress_data`의 shape이 `(T, N_nodes, 6)`일 경우 이를 `CellData`가 아닌 `PointData`로 매핑.

## Verification Plan
1. `exam4_dynamic.py`를 실행하여 SPCD에 의한 벤딩 응력을 계산합니다.
2. 결과 HDF 파일을 ParaView에서 열었을 때, `Stress (Max Envelope)`가 **Point Data**로 부드럽게 렌더링되는지 확인합니다.
3. 코너 부근의 국부적인 응력 집중(로컬 피크)이 뚜렷하게 시각화되며, 최대 응력 값이 기존(Centroid 기준)보다 크게 나타나 물리적 굽힘 거동을 제대로 모사하는지 검증합니다.
