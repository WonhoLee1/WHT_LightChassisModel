# Implementation Plan - JAX-Accelerated Dynamic Response Analysis (2026-05-12)

WHT 샤시 모델의 시간 적분(Time Integration) 과정을 JAX를 활용하여 가속화합니다. `lax.scan`을 통해 Python 루프 오버헤드를 제거하고, Sparse Iterative Solver를 사용하여 대규모 모델을 처리합니다.

## User Review Required

> [!IMPORTANT]
> - **Iterative Solver 사용**: JAX JIT 내에서는 Scipy의 SPLU(Direct Solver)와 같은 처리가 어렵기 때문에, **CG(Conjugate Gradient)**와 같은 반복법 솔버를 사용합니다. 수렴 속도를 위해 Jacobi Preconditioner를 기본 적용합니다.
> - **메모리 사용량**: JAX sparse 연산을 위해 행렬을 `BCOO` 또는 `BCSR` 형식으로 변환합니다. GPU 사용 시 메모리 한계에 유의해야 합니다.
> - **결과 정밀도**: Iterative solver의 허용 오차(`tol`)에 따라 Scipy(Direct) 결과와 미세한 차이가 발생할 수 있습니다.

## Proposed Changes

### [wht_solver] (Dynamic Solver)

#### [MODIFY] [wht_dynamic_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_dynamic_solver.py)
- `solve_direct_dynamic()`: `method` 파라미터 추가 및 분기 처리.
- `_solve_direct_dynamic_jax()` [NEW]: JAX 기반 가속 해석 로직 구현.
    - `jax.experimental.sparse`를 이용한 희소 행렬 구성.
    - `jax.lax.scan`을 이용한 시간 적분 루프 JIT 컴파일.
    - `jax.experimental.sparse.linalg.cg`를 이용한 유효 하중 방정식 풀이.
    - SPCD(처방 변위) 데이터를 JAX 배열로 변환하여 고속 보간 처리.

### [wht_topo] (Topology Optimization Tool)

#### [MODIFY] [run_topo.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/run_topo.py)
- `--solver-method` 옵션 추가 (기본값: `scipy`, 선택지: `scipy`, `jax`).
- `run_pos_dynamic()`에서 해당 옵션을 솔버에 전달.

## Verification Plan

### Automated Tests
- `python wht_topo/run_topo.py --pos-data wht_topo/sample_pos.csv --solver-method jax --no-viz` 실행.
- Scipy 결과와 JAX 결과의 Max Displacement 및 Stress 오차 비교 (1% 이내 목표).
- 전체 해석 소요 시간(Wall-clock time) 비교 및 리포트.

### Manual Verification
- ParaView를 통해 JAX 기반 해석 결과의 애니메이션 거동이 물리적으로 타당한지 확인.
