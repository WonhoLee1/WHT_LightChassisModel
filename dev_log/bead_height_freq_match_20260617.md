# 비드 높이 조정 기반 목표 모드 매칭 — 작업 기록 (2026-06-17)

## 요청 배경

WHTSolver의 고유진동수 계산 능력 + tray 메쉬에서 비드를 올려 목표 차수별 고유진동수를
맞추는 작업. 단, **토포그래피(임의 위치 비드 생성)가 아니라, 이미 비드가 형성된 위치에서
해당 영역 노드들의 높이(z_offset)만 조정**해 목표 모드 형상(MAC)과 차수별(1~3차) 고유진동수를
동시에 맞추는 것이 목적. 속도 우선순위는 "최적화 반복(iteration) 수렴 속도".

## 조사 결과

- `wht_topo/solver.py`의 기존 비드 토포 최적화 루프는 1차 모드(f1) 주파수 "크기"만 다루는
  수동 adjoint 패널티(Rayleigh quotient 근사)였고, 모드 형상 비교 로직이 없어 이번 요구에는
  맞지 않음 → 베이스로 사용하지 않음.
- 대신 `wht_solver/wht_optimizer.py` + `objectives.py` + `wht_eigensolver.py`로 구성된
  JAX 풀 오토디프 파이프라인을 베이스로 선택. `freq_loss_with_mac_assign()`이 이미 다중 모드
  freq+MAC 소프트 어사인 손실을 구현하고 있었음.
- 조사 중 발견한 미완성/버그:
  1. `DesignBounds`가 전체 노드에 동일 범위만 지원 — 비드 영역 외 노드를 고정할 수 없었음.
  2. `wht_optimizer.py`에서 `phis_opt`가 `jnp.zeros_like(...)` 더미 처리되어 MAC 항이
     실제로는 동작하지 않았음.
  3. `wht_optimizer.py:216`의 `k_args` 미정의 변수 버그 (`self._k_args`여야 함).
  4. **`DesignVariables`의 JAX pytree `unflatten`이 E/rho에 `float()`을 강제 호출** —
     `jax.grad`로 추적되는 동안 텐서(tracer)에 `float()`을 호출해 `ConcretizationTypeError`로
     무조건 크래시. 이 때문에 `WHTOptimizer.run()`은 한 번도 끝까지 실행된 적이 없었던 것으로
     보임 (사전 존재 버그, 이번 작업과 무관하게 있었음).

## 변경 사항

### `wht_solver/wht_optimizer.py`
- `DesignBounds`에 `free_node_mask: Optional[np.ndarray]` 추가 (sorted_nids 순서, False=고정).
- `clip_to_bounds()`: 고정 노드의 z_offset을 항상 0으로 강제.
- `WHTOptimizer.__init__`: `free_node_mask` 보관(`self._free_mask`).
- `WHTOptimizer.run()`:
  - 매 스텝 `grads.z_offsets *= free_mask`로 고정 노드 그래디언트 0화 (Adam 모멘텀 드리프트 방지).
  - `phis_opt`를 `freq_fn.get_last_mode_shapes()` + `stop_gradient`로 실제 형상 사용
    (모드 형상의 완전한 해석적 고유벡터 민감도는 비용이 커서 미분 경로에서 제외 — 그래디언트는
    주파수 잔차 항만 타고 흐름. 모드 교차(swap) 문제만 해결).
  - `k_args` → `self._k_args` 버그 수정.
- pytree 등록: `unflatten`에서 `float()` 강제 호출 제거 (E/rho를 0-d jnp 배열로 유지) →
  `jax.grad`가 더 이상 크래시하지 않음.

### `wht_solver/wht_eigensolver.py`
- `make_modal_freq_fn()` 내부에 클로저 변수 `_last_mode_shapes` 추가, forward 호출 시
  실제 모드 형상을 부수효과로 저장.
- `freq_fn.get_last_mode_shapes()` 헬퍼 추가.

### 신규: `wht_topo/run_bead_height_match.py`
- 비드 영역 노드셋(`*SET_NODE`)을 지정하면 해당 노드만 z_offset 자유변수로 설정.
- 두 경로:
  - `--target-model`: 별도 FEM 결과의 모드 형상까지 사용 (MAC 소프트 어사인, `WHTOptimizer`).
  - `--target-freqs`: 목표 Hz 값만 있을 때, 차수 순서를 그대로 가정한 단순 MSE 최적화
    (모드 교차가 없다고 가정할 수 있을 때 빠른 경로).

## 검증

- 합성 5x5 평판 메쉬(`scratch/smoke_test_bead_height_match.py`)로 직접 jax.grad 호출 →
  에러 없이 통과, 고정 노드 z_offset 항상 0 유지, `get_last_mode_shapes()` shape
  `(num_modes, 3N)` 정상 확인.
- **막힌 부분(환경 문제, 코드와 무관)**: `vdmc` conda 환경의 `optax`가 설치된 `jax` 버전과
  호환되지 않아 (`jax.config.update('jax_pmap_shmap_merge', ...)` → `AttributeError`)
  `import optax` 자체가 실패. `WHTOptimizer.run()`과 `run_bead_height_match.py`는 둘 다
  optax 기반이라, 패키지 버전을 맞추기 전까지는 실제 Adam 최적화 루프를 끝까지 돌려볼 수 없음.
  `pip install -U optax` 또는 호환 버전 고정이 필요 — 추후 사용자 확인 후 진행 필요.

## 참고 (계획 파일)

원본 계획 전체는 `C:\Users\GOODMAN\.claude\plans\witty-wibbling-frost.md` 참조.
