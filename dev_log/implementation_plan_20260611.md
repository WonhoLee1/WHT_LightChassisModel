# 동적 멀티 시나리오 ESL 최적화 오류 수정 및 알고리즘 무결성 검증 (2026-06-11)

이 계획서는 동적 멀티 시나리오 ESL(Equivalent Static Loads) 토포그래피 최적화 수행 시 발생하는 오류(변위/응력/변형 에너지가 0으로 출력되는 현상 및 형상 불일치 문제)와 모달 해석 시 경계조건(코너점 구속/비구속) 연속 전환으로 인한 경계조건 오염 방지 대책을 분석하고 해결하기 위한 설계 문서입니다.

## 원인 분석 (Root Cause Analysis)

1. **ESL 케이스의 과구속(Over-constraint) 발생**:
   - `wht_dynamic_solver.py`의 `extract_esl_advanced` 함수에서 동적 변위 스냅샷 `u_snap`을 이용하여 `WHTLoadCase`를 생성할 때, 모든 노드의 모든 DOF(0~5)에 대해 강제변위 경계조건(`lc.bcs`)을 추가하고 있었습니다.
   - 하지만 섀시 모델에는 RBE2 및 RBE3와 같은 다점 구속(MPC)이 포함되어 있습니다. 모든 노드의 모든 DOF에 SPC 구속(처방 변위)을 가하고 동시에 MPC 구속을 가하면 **과구속(Over-constraint)**이 발생하게 됩니다.
   - 이로 인해 정적 해석 솔버가 Augmented Stiffness System ($K_{aug}$)을 풀 때 수치적 불안정성 또는 특이 행렬(Singular Matrix) 상태에 직면하여 해가 전부 0으로 수렴하게 되었습니다. 결과적으로 변위, 응력, 변형 에너지(U, S, ...)가 모두 0으로 기록되었습니다.

2. **민감도 계산 누락 및 형상 미변화**:
   - 변위가 0으로 풀림에 따라 민감도(Sensitivity) 계산 식 $\frac{\partial C}{\partial h}$에 들어가는 변위 벡터 $u$가 0이 되어, ESL 케이스에 대한 민감도가 0이 되었습니다.
   - 따라서 전체 최적화 목적 함수에서 동적 ESL의 기여도가 사라졌고, 오직 정적 하중 케이스에 대해서만 최적화가 이루어져 비드 형상이 정하중 단독 해석 시와 완전히 동일하게 나타났습니다.

3. **올바른 ESL 방법론**:
   - ESL의 정의에 따라 동적 응답 변위 $u(t^*)$를 직접 구속 조건으로 가하는 것이 아니라, 강성 행렬과의 곱을 통해 **등가 외력(Nodal Forces)** $f_{ESL} = K u(t^*)$를 계산하여 외력 조건(`lc.forces`)으로 인가해야 합니다.
   - 이렇게 하면 과구속이 해소되고, 최적화 루프 내에서 강성 변화에 따른 정확한 변위 및 민감도 전파가 이루어집니다.

4. **모달 해석 경계조건(코너점 구속/비구속) 전환 시 원복 보장 누락**:
   - `solver.py`의 모달 해석(Iter 0 및 루프 내 모달 해석) 시 코너 노드를 임시 구속/비구속하는 과정에서 예외가 발생할 경우, 원상 복구 코드인 `fea_solver.model.spc_conditions = orig_spcs`가 실행되지 않고 건너뛰어지는 잠재적 예외 누락 버그가 존재합니다.
   - 이로 인해 원래 경계조건인 `spc_conditions`가 영구적으로 코너 구속 상태(`_temp_spcs`)로 오염되어, 그 이후에 수행되는 정적 하중 케이스들의 정해석 및 컴플라이언스 연산 결과가 크게 비틀어지거나 에너지가 0이 되는 사이드 이펙트(Side Effect)가 발생할 수 있습니다.
   - 따라서 경계조건을 임시 변경하는 모든 구간에 **`try-finally` 예외 안전 보장 구문**을 명시적으로 적용해야 합니다.

## Proposed Changes

### 1. [wht_solver] [wht_dynamic_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_dynamic_solver.py)
`extract_esl_advanced` 메서드 내에서 변위를 경계조건(`bcs`)으로 가하던 로직을 등가 외력(`forces`)을 계산하여 인가하는 방식으로 전면 수정합니다.

- **기존 코드**:
  ```python
  lc = WHTLoadCase(name=f"{prefix}ESL_{esl_idx+1:02d}_t{t_val:.4f}s_SE{se_val:.1e}")
  for i, nid in enumerate(sorted_nids):
      u_node = u_snap[i]
      lc.add_bc(nid, dofs=(0, 1, 2, 3, 4, 5), value=0.0)
      for d in range(6):
          val = float(u_node[d])
          if abs(val) > 1e-15:
              lc.bcs.append(WHTBCEntry(nid, (d,), val))
  ```
- **변경할 코드**:
  ```python
  u_flat = u_snap.flatten()[:ndof]
  f_esl = K @ u_flat  # 등가 외력 계산
  
  lc = WHTLoadCase(name=f"{prefix}ESL_{esl_idx+1:02d}_t{t_val:.4f}s_SE{se_val:.1e}")
  for i, nid in enumerate(sorted_nids):
      f_node = f_esl[i * 6 : i * 6 + 6]
      if np.max(np.abs(f_node)) > 1e-12:
          lc.forces.append(WHTForceEntry(nid, tuple(float(v) for v in f_node)))
  ```

### 2. [wht_topo] [solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/solver.py)
`normalize_obj` 옵션이 켜져 있을 때 동적 ESL 케이스 이름의 미세한 변경(시점 $t$나 에너지 $SE$ 값 변경)에도 초기(Flat) 상태의 기준 컴플라이언스($C_{i0}$)를 정확하게 매핑할 수 있도록 `get_C0_safe` 함수를 고도화합니다.

- **변경할 코드**:
  ```python
  def get_C0_safe(n):
      if n in C0_cases:
          return C0_cases[n]
      # 접두사 패턴(예: 'rear_ESL_01' 또는 'ESL_01')을 정규식으로 추출하여 매칭
      import re
      match = re.search(r'(.*ESL_\d+)', n)
      if match:
          base = match.group(1)
          for old_n, old_val in C0_cases.items():
              if old_n.startswith(base):
                  return old_val
      # 기존 폴백 유지
      if '_t' in n:
          base = n.split('_t')[0]
          for old_n, old_val in C0_cases.items():
              if old_n.startswith(base + '_t'):
                  return old_val
      return 1.0
  ```

### 3. [wht_topo] [solver.py (고유진동수 경계조건 원복 보장)](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/solver.py)
Iter 0 및 루프 내 고유진동수 해석 시, 경계조건을 임시 구속 상태로 변경했다가 원복하는 로직에 `try-finally` 예외 복구 구문을 적용하여 경계조건이 복구되지 않고 오염되는 현상을 완벽하게 방지합니다.

- **변경할 코드 (Iter 0 모달 해석)**:
  ```python
  _orig_spcs = list(fea_solver.model.spc_conditions)
  try:
      _spcd_nids = [nid for nid in fea_solver.model.nodes if nid >= 900000]
      _temp_spcs = [WHTSPCEntry(nid, (0,1,2,3,4,5), 0.0) for nid in _spcd_nids]
      fea_solver.model.spc_conditions = _temp_spcs if _spcd_nids else _orig_spcs
      _ref_modal = fea_solver.solve_modal(
          num_modes=self.modal_modes, exclude_rigid_body=False
      )
      _ref_freqs = _ref_modal.frequencies.tolist()
  finally:
      fea_solver.model.spc_conditions = _orig_spcs
  ```

- **변경할 코드 (루프 내 고유진동수 해석)**:
  ```python
  orig_spcs = list(fea_solver.model.spc_conditions)
  try:
      _spcd_nids = [nid for nid in fea_solver.model.nodes if nid >= 900000 or nid in corner_nids]
      _temp_spcs = [WHTSPCEntry(nid, (0,1,2), 0.0) for nid in _spcd_nids]
      fea_solver.model.spc_conditions = _temp_spcs if _spcd_nids else orig_spcs
      modal_results = fea_solver.solve_modal(num_modes=self.modal_modes, exclude_rigid_body=False)
      freqs = modal_results.frequencies
  finally:
      fea_solver.model.spc_conditions = orig_spcs
  ```

## 추가 검토 사항: 2단계 연속 모달 해석(구속/비구속) 수행 시의 잠재적 문제점 및 무결성 대책

1. **강체 모드 필터링(`exclude_rigid_body`) 오작동**:
   - 비구속(Free-Free) 모달 해석 시에는 첫 6개의 강체 모드(0Hz 근처)를 걸러내기 위해 `exclude_rigid_body='auto'` 또는 `'skip6'`가 올바르게 작동해야 합니다.
   - 그러나 코너점 구속(Constrained) 모달 해석을 수행할 때는 강체 모드가 존재하지 않고 1차 모드부터 유효한 탄성 모드가 됩니다.
   - 이 두 해석을 연속으로 수행하면서 동일한 필터 설정을 적용하면, 구속 모달 해석의 1~6차 탄성 모드가 강제 생략(skip)되거나 강체 모드로 오인되어 필터링되는 데이터 왜곡이 발생합니다.
   - **대책**: 각 해석 성격에 맞추어 `exclude_rigid_body` 옵션을 구속 시에는 `False`로, 비구속 시에는 `'auto'` 또는 `'skip6'`로 명확히 분기 제어하여 모드 왜곡을 방지합니다.

2. **수치적 수렴 지연 및 ARPACK 데드락**:
   - 구속 모달 해석의 첫 번째 탄성 고유진동수가 수십~백 Hz 이상으로 큽니다.
   - 이때 비구속 해석용의 기본 음수 shift 값($\sigma = -1.0$)을 그대로 공유하게 되면, ARPACK 솔버가 타겟 고유치와 멀리 떨어진 영역에서 수렴하기 위해 지나치게 많은 이터레이션을 돌며 수렴에 실패하고 데드락에 빠지기 쉽습니다.
   - **대책**: 구속 모달 해석 시에는 shift-invert 파라미터 $\sigma$를 0.0 또는 양수 값으로 설정하고, 비구속 시에는 $\sigma \approx -1.0$을 사용하도록 개별 제어하여 수치 해석적 강건성을 보장합니다.

3. **Rayleigh 감쇠 계수 산출을 위한 eigsh 중복 실행 병목**:
   - `solve_direct_dynamic`에서 감쇠비 ζ를 감쇠 행렬 C로 변환할 때 내부적으로 `_rayleigh_coeffs`가 호출되어 `eigsh`를 통해 저차 주파수를 구합니다.
   - 2단계 연속 해석 시 구속조건이 급격히 바뀔 때, 이 내부 `eigsh` 역시 $\sigma = -1.0$으로 하드코딩되어 있어 구속 모델에서 수렴이 실패하고 기본 디폴트 주파수 폴백으로 대체되어 감쇠비 자체가 찌그러지는 현상이 발생합니다.
   - **대책**: Rayleigh 감쇠 계산용 고유치 해석 시에도 모델의 구속상태에 맞는 적절한 shift 파라미터를 넘겨주거나, 사전 모달 결과를 안전하게 연계하는 예외 제어가 필요합니다.

4. **JAX 가속 동해석 (`_solve_direct_dynamic_jax`) 파티셔닝 경계부 오버헤드**:
   - n_free 임계값(10000)을 기준으로 Dense JAX LU와 SciPy Sparse 솔버로 자동 분기됩니다.
   - 코너점 구속 시와 비구속 시에 자유도가 10000 부근에서 걸치게 되면 한 루프 내에서 JAX JIT 컴파일과 SciPy Sparse 계산이 매번 교대로 트리거되어 Tracing 오버헤드 및 GPU/CPU 간 메모리 이동 지연이 발생할 수 있습니다.
   - **대책**: JIT 컴파일이 반복 호출되는 것을 방지하기 위해, 한 실행 컨텍스트 내에서는 파티셔닝 솔버 분기 옵션을 강제 고정하는 옵션이 요구됩니다.

---

## [추가 수정 계획] Z방향 강체 거동(Rigid Body Motion) 발산 및 정하중 누락 분석 (보강)

1. **Z방향 강체 거동 발산 해결 (3-2-1 정정 구속조건 도입)**:
   - 원인: 동적 변위 $u(t)$로부터 계산된 등가 외력 $f_{ESL} = K u$를 정적 해석으로 풀 때, Z방향 및 회전 자유도 구속이 없어 수치적으로 무한히 발산($10^{11}$ mm)하는 치명적인 문제가 확인되었습니다.
   - 해결 방안: `run_topo.py` 내부의 `_build_snapshots` 및 `_build_element_peak_lc`에서 3-2-1 정정 구속조건(C5 코너: XYZ, C6 코너: Z, C8 코너: Z)을 각 하중케이스의 경계조건(SPC)으로 명시적으로 추가 적용하도록 수정합니다.
   - 기대 효과: 강체 거동이 완벽히 억제되어 실제 동적 변형 크기(수 mm) 및 Nodal Stress(수백 MPa) 범위 내로 정적 해석 결과가 수렴하게 됩니다.

2. **정하중 케이스 누락 현상 분석**:
   - 분석 결과, 사용자가 이전 실행 시 `--no-static` 플래그를 지정하여 정적 하중 케이스가 제외되었음이 확인되었습니다. 코드 상의 누락 결함은 아닙니다.
   - 또한 기존에는 ESL 하중의 발산 수치($10^{11}$ mm)와 정하중 변형(수 mm)의 척도 차이가 너무 커서 정하중 그래프가 보이지 않았을 수 있습니다. 3-2-1 구속으로 스케일을 복원하면 정하중과 ESL 하중의 스케일이 맞춰져 모두 시각적으로 정상 표현될 것입니다.

3. **WHTopographySolver 모니터 설정 데이터 점검**:
   - `solver.py`의 `_get_run_settings`에서 모니터 GUI에 전달하는 옵션 딕셔너리(`max_iter`, `h_max`, `bead_area`, `min_width` 등)가 정상적으로 전달되는지 다시 한 번 점검합니다.

## Verification Plan

### Automated Tests
- CLI 환경에서 테스트 명령어를 수행하여 3-2-1 정정 구속조건이 잘 인가되는지 로그를 검증합니다.
  `& C:\Users\GOODMAN\miniconda3\envs\vdmc\python.exe wht_topo\run_topo.py --ignore-gooey --dynamic-opts wht_topo\structural_dynamics_rear.csv --iters 2 --no-gui --no-viz`

### Manual Verification
- 이터레이션 완료 후 `results/최신디렉토리/iter_stats.json` 파일의 `wht_topo_ESL_01...` 케이스 결과에서 변위 및 응력이 수십 mm / 수백 MPa 수준의 물리적으로 타당한 범위에 안착하는지 수치적으로 확인합니다.
- 모니터링 UI의 Settings 테이블(파라미터, 값, 단위, 설명) 탭 데이터가 정상적으로 수집되어 출력되는지 검수합니다.
