# Walkthrough: Shell Analysis Stabilization & Static Pipeline

쉘 요소의 수치적 불안정성을 해결하고, 정적 하중 해석을 통한 강성 검증 파이프라인을 구축했습니다.

## 1. 해결된 주요 문제 (Hotfixes)

### 쉘 요소 수치적 하드닝 (Numerical Hardening)
- **드릴링 패널티(Drilling Penalty)** 강화: 비정형 메시에서의 조건수 개선을 위해 $0.01 \times G \times t$로 상향.
- **회전 관성 하한치(Inertia Floor)** 설정: $1e^{-8}$을 적용하여 ARPACK(고유치 해석기)의 수치적 정지(Hang) 현상 해결.

### Solver 안정성 및 정합성 보강
- **강성 조립 통합**: `solve_modal`과 `solve_static`에서 사용하는 강성 조립 로직을 `_assemble_K_scipy`로 일원화하여 `QUAD4` 강성 누락 버그 수정.
- **AUTOSPC 적용**: 정적 해석 시에도 미세 강성 자유도를 자동으로 보강하여 비정형 메시에서의 특이행렬(`MatrixRankWarning`) 문제 해결.
- **SciPy Solver 전환**: JAX 타입 호환성 이슈를 피하기 위해 정적 해석에 `scipy.sparse.linalg.spsolve` 직접 사용.

## 2. 정적 하중 해석 검증 결과

`exam2_shell_jaxSSO_load.py`를 통해 1000N 중앙 집중 하중 조건에서의 쉘 트레이 변위를 분석했습니다.

| Mesh Type | Nodes | Max Uz (mm) | Diff (%) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **QUAD4** | 1876 | 2451.68 | 0.00% | **Success** |
| **TRIA3** | 1876 | 1544.32 | -37.01% | **Success** |
| **MIXED** | 1876 | 2456.45 | +0.19% | **Success** |
| **TRIA3_FREE**| 1971 | 2058.24 | -16.05% | **Success** |

> [!NOTE]
> `TRIA3` 요소가 `QUAD4` 대비 약 37% 단단하게 계산되는 전형적인 Locking 현상이 관찰되나, 비정형 메시(`TRIA3_FREE`)에서도 안정적으로 수렴함을 확인했습니다.

## 3. 관련 파일 및 경로

- **Solver**: [wht_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_solver.py) (리팩토링 완료)
- **Visualizer API**: [wht_visualizer.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_visualizer/wht_visualizer.py) (호환성 함수 추가)
- **검증 스크립트**: [exam2_shell_jaxSSO_load.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/exam2_shell_jaxSSO_load.py) (신규)

## 4. 수치적 제언
- 현재 $0.6mm$ 박판에 $1000N$ 하중은 기하학적 비선형성(Large Deflection)이 지배적인 영역입니다. 선형 강성 비교를 위해서는 하중을 $10N$ 정도로 낮추어 재검증하는 것을 권장합니다.
- 비정형 메시의 정확도를 높이기 위해 차기 단계에서 `TRIA6`(6절점 삼각 요소) 도입을 검토할 수 있습니다.
