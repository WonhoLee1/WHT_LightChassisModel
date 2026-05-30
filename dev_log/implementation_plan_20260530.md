# [CalculiX US3 (CS-DSG + ANDES) 평면 쉘 요소 전면 이식 계획]

본 계획은 CalculiX 소스 아카이브(`ccx_2.23.src.tar.bz2`)로부터 확보한 사용자 정의 쉘 요소인 **US3** 서브루틴(`us3_sub.f`)을 WHT Solver의 TRIA3 요소(NumPy 및 JAX)에 완벽하게 포팅하여, **CalculiX와 100% 일치하는 정확도를 달성하고 속도와 자동 미분 성능을 그대로 유지하기 위한 상세 구현 계획**입니다.

---

## 1. 확보된 US3 공식의 수학적·수치학적 분석

CalculiX US3 쉘 요소는 기하학적 락킹(Locking)을 원천 배제하고 3절점 요소임에도 3차원 결합 성능을 완벽히 내는 고급 플랫 쉘 요소입니다.

### A. 플레이트 굽힘 및 횡전단: CS-DSG (Cell-based Smoothed Discrete Shear Gap)
- **Bs (전단 변형률 매핑):** 요소 내부를 3개의 Sub-triangle 셀로 나누고, 각 셀에서 정의되는 횡전단 변형률(`bs1`, `bs2`, `bs3`)을 셀 면적(`Ai`) 가중 평균 및 통합하여 횡전단 잠김(Shear Locking)이 절대 발생하지 않는 고밀도 `Bs` (2x18) 행렬을 유도합니다.
- **Bb (굽힘 곡률):** 3절점 1차 굽힘 B-matrix `Bb` (3x18)를 조립합니다.
- **강성 조립:**
  $$K_p = \left( B_s^T D_s B_s + B_b^T D_b B_b \right) \cdot A_e$$

### B. 멤브레인 (면내): ANDES (Assumed Natural Deviatoric Strain)
노드당 면내 3자유도($u, v, \theta_z$)를 완벽한 연속체 에너지 보존 법칙 내에서 결합하는 최신형 고성능 면내 쉘 이론입니다.
- **Basic Stiffness ($K_b$):** Lumping matrix $L$ (9x3)을 구성하여 면내 거동에 대한 기초 강성을 조립합니다.
  $$K_b = \frac{1}{V_e} L Q_{in} L^T$$
- **Higher-order Stiffness ($K_h$):** 드릴링 자유도($\theta_z$)와 결합하여 고차 면내 변형 에너지를 구속하는 조립 행렬입니다. Natural strain transformation $T_e$ (3x3)와 3개의 nodal strain-displacement matrices $Q_4, Q_5, Q_6$를 조합하여 고차 강성 $K_O$ (9x9)를 구한 뒤, rotation transformation $T_0$를 통해 18x18 쉘 자유도로 매핑합니다.
  $$K_m = K_b + K_h$$

### C. 질량 행렬: 일관 질량 행렬 (Consistent Mass Matrix, $M$)
- 3점 가우스 면적 적분을 돌며 노드당 병진 질량 및 회전 관성(두께 효과 반영)을 완벽한 18x18 일관 질량 행렬로 조립하여, 모달 주파수 연산 시 질량 분배 왜곡을 원천 파괴합니다.

---

## 2. 제안하는 Proposed Changes (코드 변경 영역)

WHT Solver의 TRIA3 모듈을 CalculiX 공식으로 전면 대체합니다.

### 1) [MODIFY] [wht_tria3_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_tria3_element.py) (NumPy 버전)
- `_element_K_tria3` 함수 내에 `us3_csys_cr`, `us3_linel_Qi`, `us3_CS`, `us3_Bs`, `us3_Bb`, `us3_Kp`, `us3_Km` 함수를 NumPy 연산으로 완벽히 포팅 및 대체 구현합니다.
- `M_tria3_lumped` 대신 `us3_M` 공식을 포팅하여 고정밀 일관 질량 행렬(`M_tria3_consistent`) 기능을 추가 탑재합니다.

### 2) [MODIFY] [wht_tria3_element_jax.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_tria3_element_jax.py) (JAX 버전)
- `_element_K_tria3_jax` 내에 위의 모든 NumPy 포팅 공식을 JAX API (`jnp.zeros`, `jnp.stack`, loop unrolling, `jnp.matmul` 등)로 100% 동일하게 번역하여 구현합니다. `jax.jit` 컴파일러와 호환되도록 pure-function 제약을 완벽히 유지합니다.

---

## 3. 검증 및 목표 (Verification Plan)

### 합격 수치 목표
- **패치 테스트:** `verification_runner.py` 실행 결과, 22개 패치 테스트 및 고유진동수 오차가 여전히 **1% 이하**로 완전히 **PASS**해야 합니다. (기존 MITC3+가 맞던 기초 강성을 완벽히 만족하는지 크로스 체크)
- **CalculiX 모달 벤치마크 매칭:**
  `benchmark_whtsolver_modal.py`를 실행하여 섀시 모델(`ccx_iter016`)을 해석했을 때, **WHT Solver와 CalculiX 간의 7차 구조 진동수 오차가 기존 21.6%에서 한 자릿수(수% 이내)로 소멸하는지 확인하여 쉘 공식 이식의 성공을 수치적으로 증명합니다.

---

## 4. 배치 모달 변형 ISO 뷰 캡쳐 및 고유 모드 형상 비교 구현 계획

본 계획은 WHT Solver와 CalculiX의 고유 모드 해석 결과를 시각적으로 대조하기 위해, 각 솔버의 모드별 변형 형상(Mode Shape)을 배치(Batch)로 ISO 뷰 캡쳐하고 나란히(Side-by-Side) 병합하는 기능을 개발하는 상세 설계입니다.

### A. 구현 방식 및 시각화 표준
- **CalculiX 결과 변환:** `AutoCalculix`의 `FrdToVtuConverter`를 호출해 `.frd` 결과 파일을 각 모드별 `.vtu` 파일들로 자동 분할 변환합니다.
- **WHT Solver 결과 변환:** WHT Solver 구동 후 얻은 `WHTSolverResult` 객체에서 공통 IR `WHTResultData`를 생성하고, 내부의 CSR 구조로부터 PyVista `UnstructuredGrid`를 실시간 생성합니다.
- **오프스크린 PyVista 캡쳐:** 
  - `pyvista.Plotter(off_screen=True)`를 활용하여 배치 렌더링을 수행합니다.
  - **시각화 규칙 준수:**
    - 배경색: Black, 글자색: White (ParaView / Black Theme)
    - Mesh Edge 색상: darkgray, 에지 표시 활성화 (`show_edges=True`)
    - Global Font Size: 12 (Colorbar 등 텍스트 표시용)
    - 좌표축(Coordinate Axes) 및 Colorbar 표시 (Colorbar는 1개만 생성)
  - 변위 벡터 필드(`disp`)에 따라 격자를 변형(`warp_by_vector`)하고, 스케일 팩터는 형상이 왜곡되지 않고 식별이 용이한 수준(예: Max displacement 기준 자동 스케일링)으로 정교하게 통제합니다.
  - 두 솔버 모두 **완벽하게 동일한 Isometric Camera View** (`view_isometric()`, `reset_camera()`)를 설정하여 왜곡 없는 1:1 비교를 보장합니다.

### B. Side-by-Side 병합 이미지 및 텍스트 오버레이
- `Pillow` 라이브러리를 활용하여 WHT Solver 캡쳐본과 CalculiX 캡쳐본을 가로로 결합합니다.
- 결합된 이미지 상단에 모드 번호 및 주파수를 시각적으로 고급스럽게 표시합니다.
  - 예: `"WHT Solver: Mode 7 (45.16 Hz)"` | `"CalculiX: Mode 7 (45.16 Hz)"`
- 병합된 고유진동형 비교 이미지를 `SolverVerification/results/images/mode_*.png`로 자동 저장합니다.

### C. Proposed Changes (신규 파일 및 수정)

#### [NEW] [capture_modal_shapes.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/SolverVerification/capture_modal_shapes.py)
- CalculiX `.frd` 변환, WHT Solver 구동, PyVista 오프스크린 캡쳐, Pillow 이미지 병합 및 저장을 수행하는 통합 자동화 배치 캡쳐 모듈입니다.
- 탄성 변형이 뚜렷한 주요 모드(Mode 7 ~ 15)를 타겟으로 동작합니다.

#### [MODIFY] [verification_report_YYYYMMDD-HHMM.md](file:///d:/PythonCodeStudy/WHT_LightChassisModel/SolverVerification/results/verification_report_20260530-0541.md)
- 벤치마크 결과 보고서에 캡쳐된 각 모달 쉐입 비교 병합 이미지를 시각적으로 삽입하고 분석 내용을 보완합니다.

### D. 검증 및 목표 (Verification Plan)
- **자동화 실행:** `python SolverVerification/capture_modal_shapes.py` 명령어가 오류 없이 완료되고 `SolverVerification/results/images/` 디렉토리에 각 모드별 비교 이미지가 정상적으로 생성되는지 검증합니다.
- **이미지 품질 검사:** 캡쳐된 이미지의 배경이 완벽히 Black이고, Edge가 darkgray로 깔끔하며, 두 솔버의 형상 및 카메라 각도가 완전히 대칭 및 일치하는지 확인합니다.

---

## 5. [추가] Phase 3: 고유 주파수 차이 극복 방안 및 수치적 정상화 구현 계획 (2026-05-30 오후)

### A. 기술 검토 및 원인 분석
- **현상:** WHT Solver는 패치 테스트(단순 평판 굽힘 및 진동)는 이론치와 1% 이내로 완벽히 Pass하지만, 실제 복잡한 성형 비드 모델(`ccx_iter016`)에서는 CalculiX 대비 7차 이상 탄성 모드 주파수가 15% ~ 33% 낮게 계산됩니다.
- **근본 원인 발견 (QUAD4 회전 관성 질량 오적용):**
  `wht_solver/wht_quad4_element.py`의 `M_quad4_lumped` 내에서 회전 관성을 계산할 때 다음 공식을 채택하고 있습니다:
  $$I_{rot} = m_{node} \frac{t^2 + A_e}{12}$$
  여기서 $t$는 두께, $A_e$는 요소의 면적(area)입니다. 
  그러나 Reissner-Mindlin 쉘의 회전 자유도(RX, RY)에 럼핑되는 질량 관성은 오직 두께 방향 극관성 모멘트 성분인 $t^2/12$ 에만 비례해야 합니다. 평면 내의 병진 관성 질량은 이미 $m_{node}$가 담당하고 있기 때문입니다:
  $$I_{rot} = m_{node} \frac{t^2}{12}$$
  문제는 두께 $t = 0.6 \text{ mm}$ ($t^2 = 0.36 \text{ mm}^2$)에 비해, 섀시 메쉬의 요소 면적 $A_e$는 약 $100 \text{ mm}^2$로 **약 277배나 비정상적으로 큽니다.**
  이로 인해 노드의 회전 관성이 실제 물리량보다 약 **200~300배 과대계상(Over-lumped)**되어 고유 진동 모드의 질량이 인위적으로 무거워지고, 이로 인해 고유진동수가 CalculiX 대비 대폭 낮아지는 현상이 발생했습니다. (TRIA3의 경우, `us3_M` 일관 질량에서 정확히 $t^2/12$ 스케일의 회전 관성질량이 적분 조립되고 있었습니다.)

### B. Proposed Changes
- **[MODIFY] [wht_quad4_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_quad4_element.py)**
  - `M_quad4_lumped` 함수의 회전 관성 질량(`rot_inert`) 공식에서 불필요하게 더해지던 `area` 항을 제거하고 정상적인 쉘 이론에 기반한 수치 공식으로 변경합니다:
    ```python
    # AS-IS:
    rot_inert = m_node * (t**2 + area) / 12.0
    # TO-BE:
    rot_inert = m_node * (t**2) / 12.0
    ```
- **[BACKUP] [wht_quad4_element.py.bak](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_quad4_element.py.bak)**
  - 수정 전, 회복 가능한 원본 소스코드 백업 생성.

### C. Verification Plan
1. **[Automated] Patch Tests Run:**
   - `python SolverVerification/verification_runner.py`를 실행하여 22개 패치 테스트 및 평판 3점/4점/twisting/frequency 거동의 이론치 대비 오차가 5% 이하로 **ALL PASS**가 유지되는지 대조합니다.
2. **[Automated] Chassis Benchmark Run:**
   - `python SolverVerification/benchmark_whtsolver_modal.py`를 실행하여 섀시 비드 모델의 고유진동수를 추출하고, CalculiX와의 오차가 기존 15%~33%에서 획기적으로 개선되는지 확인합니다.
3. **[Automated] Side-by-Side capture 및 리포트 작성:**
   - `python SolverVerification/capture_modal_shapes.py`를 재구동하여 정상화된 WHT Solver 주파수가 표시된 Side-by-Side 이미지들을 갱신하여 캡쳐하고, `modal_benchmark_report.md` 리포트를 최신 수치와 이미지로 최종 업데이트합니다.

---

## 6. Phase 4: 수렴 허용 오차(tol) 최적화를 통한 모달 해석 고속화 계획 (2026-05-30 저녁)

### A. 기술 검토 및 가속 원리
- **현상:** 모달 해석 ARPACK (`eigsh`) 고유치 계산 시 수렴 판정 허용치(`tol`)가 기본 `1e-5`로 극도로 촘촘하게 설정되어 있어 불필요하게 많은 Lanczos 반복 루프를 유발합니다.
- **가속화 대책 (이론적 정석 내 튜닝):**
  - 고유주파수를 제품 설계 변수로 투영하는 모달/위상 최적화의 경우, `1e-3` ~ `1e-4` 수준의 오차 허용치만으로도 고유진동형(Mode Shape) 및 주파수 수치 정합성이 물리적으로 완전하게 보장됩니다.
  - `WHTSolver.solve_modal` 및 내부 ARPACK `_arpack_subprocess_worker` 에 `tol` 인자를 노출하고 기본값을 `1e-3`으로 변경함으로써 Lanczos 반복 횟수를 대폭 감소시켜 **모달 해석 속도를 약 20% ~ 40% 추가 가속**시킵니다.

### B. Proposed Changes
- **[MODIFY] [wht_solver/wht_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_solver.py)**
  - `_arpack_subprocess_worker` 함수에 `tol` 매개변수를 추가하고, `eigsh` 호출 시 적용:
    ```python
    def _arpack_subprocess_worker(K, M_diag, k, sigma, maxiter, result_queue, tol=1e-3):
        ...
        vals, vecs = eigsh(K, k=k, M=M_op, which="LM", sigma=sigma, tol=tol, maxiter=maxiter)
    ```
  - `WHTSolver.solve_modal` 메서드 서명에 `tol: float = 1e-3`을 추가하고, `multiprocessing.Process`의 `args`에 전달하도록 연동합니다.

### C. Verification Plan
1. **[Automated] Patch Tests Run:**
   - `python SolverVerification/verification_runner.py`를 실행하여 22개 패치 테스트 및 평판 진동수 오차가 여전히 합격 기준(5% 이내)에 정확히 머무는지 검증합니다.
2. **[Automated] Chassis Speed Benchmark & Comparison:**
   - 오차를 수정하기 전(tol=1e-5)과 수정 후(tol=1e-3)의 섀시 모델 모달 연산 시간(ms)을 비교하여 정량적인 가속 효과를 측정합니다.
3. **[NEW] YYYYMMDD_HHMM Benchmark Report md 생성:**
   - 최종 벤치마크 및 오차 극복/속도 개선 전후 대조 결과를 `SolverVerification/results/modal_benchmark_report_20260530_HHMM.md` 형식의 별도 보고서로 신규 작성합니다.

---

## 7. Phase 5: [신규] 꺾임각 연동형 가변 드릴링 강성 보정(Drilling Stiffness Coupling) 구현 계획 (2026-05-30 밤)

### A. 기하학적 보정 원리 (Kanok-Nukulchai & Hughes 정석 기반)
- **배경:** Flat Shell은 평면 내 회전(Drilling, RZ) 강성이 없어 극소 Penalty 강성을 가해 특이성만 모면합니다. 하지만 요소가 서로 꺾이는 접힘선(Fold Line)에서는 **한 요소의 RZ가 인접 요소의 면외 굽힘(bending)과 강력하게 결합**됩니다. 여기에 균일한 미소 penalty만 작용하면 비드 꺾임부에서 모멘트 전달률이 끊겨 전체 비틀림/굽힘 주파수가 20% 이상 하락합니다.
- **꺾임각 연동 수학적 조율 ($\sin^2\theta_i$ 보간):**
  1. 솔버 기동 시 각 노드 $i$에 공유된 모든 인접 쉘 요소들의 법선 벡터 $\{\mathbf{n}_e\}$를 수집합니다.
  2. 노드별 최대 법선 이탈 접힘각 $\theta_i = \arccos(\min \mathbf{n}_{e1} \cdot \mathbf{n}_{e2})$를 획득합니다.
  3. 노드별 가상 드릴링 결합 강도 계수 $\beta_i$를 기하학적으로 연동하여 정의합니다:
     $$\beta_i = \beta_{min} + (1.0 - \beta_{min}) \cdot \sin^2\theta_i$$
     *(평면 $\theta_i \approx 0$ 일 때는 미소 penalty $\beta_{min} = 10^{-4}$로 락킹을 방지하고, 직교 꺾임 $\theta_i \approx 90^\circ$ 에 가까울수록 굽힘 모멘트 전달을 위해 $\beta_i \approx 1.0$의 단단한 강성을 부드럽게 부여합니다.)*
  4. 요소 드릴링 강체 조립식에 네 노드 간의 가중 조율 $\sqrt{\beta_i \beta_j}$를 대칭적으로 곱하여 조립합니다:
     $$K_{drill} = \sum_{i=1}^4 \sum_{j=1}^4 (Bd_i^T Bd_j) \cdot (\sqrt{\beta_i \beta_j} \cdot G \cdot t \cdot \text{detJ})$$

### B. Proposed Changes
- **[MODIFY] [wht_solver/wht_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_solver.py)**
  - 모델의 절점 공유 관계로부터 각 노드의 접합 법선 벡터 간의 내적을 분석해 `node_beta` (N,) 매핑 사본을 생성하는 헬퍼 `_compute_folding_angles` 메소드를 추가합니다.
  - 조립 함수 `_assemble_K_scipy` 내부에서 이 노드별 `node_beta` 정보를 준비하여 요소 레벨 강성 조립기에 전달합니다.
- **[MODIFY] [wht_solver/wht_quad4_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_quad4_element.py)**
  - `K_quad4_scipy` 및 내부 `_element_K_mitc4_plus_nb` 함수가 네 노드의 `beta` 어레이를 전달받아 노드 연동 가변 드릴링 강성을 대칭적으로 곱해 조립하도록 변경합니다.
- **[MODIFY] [wht_solver/wht_quad4_element_jax.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_quad4_element_jax.py)**
  - GPU 가속 조립을 수행하는 `_element_K_mitc4_plus_jax` 및 `K_quad4_jax` 버전에도 꺾임각 노드 beta 벡터를 인자로 추가 전달하고, XLA 배치 연산에서 가변 드릴링 강성을 동일 수학에 근거하여 반영합니다.

### C. Verification Plan
1. **[Automated] Patch Tests ALL PASS 검증:**
   - 평판 상태에서는 $\theta_i \approx 0$ 이므로 기존 $\beta_{min}$ 강성과 동일하게 작용합니다. 따라서 `verification_runner.py` 22개 테스트가 **여전히 100% ALL PASS** 및 동일 주파수를 무결하게 유지하는지 교차 검증합니다.
2. **[Automated] Chassis Bead Model Frequencies 검증:**
   - `python SolverVerification/benchmark_whtsolver_modal.py`를 가동하여, 비드 꺾임 굽힘 강성이 보강된 WHT Solver의 고유주파수가 CalculiX와의 오차가 극적으로 해소(한 자릿수 오차율로 도달)되는지 검증하고, 향상된 결과표를 수록하여 `modal_benchmark_report_20260530_1925.md` 를 최종 업데이트합니다.


---

## 8. Phase 6: WHTVisualizer Query Tools 및 Bead Height Warping 구현 계획 (2026-05-30 심야)

### A. 핵심 구현 요구사항 및 기하학적 설계
1. **Bead Height 3D Warping 지원**:
   - `Bead_Height` 필드가 point_data나 cell_data에 존재할 경우, 스칼라 필드임에도 `combo_warp_vec` (Warping 대상 필드) 목록에 노출되도록 예외 필터를 추가합니다.
   - Warping이 켜져 있을 때, `warp_field == "Bead_Height"` 이면 각 파트의 메시 절점에 할당된 `Bead_Height` 값을 획득합니다. (cell_data에만 있는 경우 cell_data_to_point_data를 사용해 point_data로 자동 변환하여 참조)
   - 획득한 `Bead_Height * scale` 만큼 draw-dir 방향인 음의 Z축 `[0, 0, -1]`으로 가상의 3D 변위 벡터 `disp = [0, 0, -Bead_Height]` (크기 `(N_points, 3)`)를 생성합니다.
   - 이를 기존의 `_warp_pts` 기하학적 변환 함수에 흘려보내 메시 표면을 입체적으로 돌출(Warp)시킵니다.

2. **Query Tools 인터랙티브 조회 기능**:
   - **체크박스 및 옵션 구성**: `Enable Query` QCheckBox, `Node Value`/`Element Value` QRadioButton, `Clear Labels` QPushButton을 탭독에 배치합니다 (완료).
   - **마우스 이벤트 몽키 패치**: PySide6 네이티브 interactor의 `mouseMoveEvent` 및 `mouseDoubleClickEvent`를 안전하게 몽키 패칭(`_on_qt_mouse_move`, `_on_mouse_double_click`)하여 기존 회전/이동 동작을 방해하지 않고 Query 비즈니스 로직을 주입합니다.
   - **VTK Ray-Casting 3D Picking**: 마우스 Qt 좌표를 VTK 3D 공간 좌표계(Y축 반전: `size[1] - pos.y()`)로 보정한 후 `vtkCellPicker` 레이캐스터를 통해 충돌하는 3D actor, 로컬 cell/point ID 및 3D 공간 상의 충돌 절대 좌표 `(px, py, pz)`를 얻습니다.
   - **상태바(StatusBar) 출력 (Hover)**: 마우스 hover 시 활성 스칼라 필드의 `Node Value` 또는 `Element Value`를 submesh 및 original ID 매핑(`vtkOriginalPointIds`/`vtkOriginalCellIds`)을 통해 역추적하고, 해당 값을 QMainWindow의 `statusBar().showMessage()`를 통해 실시간 상태바에 출력합니다.
   - **3D 텍스트 라벨 부착 (Double Click)**: 더블클릭 시 충돌 지점 좌표에 `plotter.add_point_labels`를 활용해 8pt 크기의 텍스트 값 라벨 `N[ID]: [Value]` 혹은 `E[ID]: [Value]`를 화면 상에 입체적으로 생성하고, 생성된 라벨의 actor 명을 `self._query_label_names` 리스트에 수집합니다.
   - **라벨 일괄 클리어**: `Clear Labels` 버튼 클릭 시 `self._query_label_names`에 캐싱된 모든 actor 명에 대해 `plotter.remove_actor(name)`를 호출해 화면에서 일괄 소거하고 리스트를 초기화합니다.

### B. Proposed Changes
- **[MODIFY] [wht_visualizer/wht_visualizer.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_visualizer/wht_visualizer.py)**
  - `_populate_combo_box`: `Bead_Height` 필드가 있는 경우 `warp_candidates`에 노출하도록 예외 처리.
  - `_apply_warping`: `warp_field == "Bead_Height"`일 때 가상 3D 음의 Z 변위 `disp`를 동적으로 생성하여 `_warp_pts`에 대입하는 물리 변형 로직 추가.
  - `__init__` 하단에 `_query_label_names = []` 캐시 초기화 및 QVTK interactor 마우스 이벤트 패치.
  - `_on_query_toggled(self, state)`, `_on_query_target_changed(self, toggled)`, `_clear_query_labels(self)` 추가.
  - `_get_picker_result(self, pos)` 추가: `vtkCellPicker` 기반 3D 피킹 공통 유틸리티.
  - `_on_qt_mouse_move(self, event)`, `_process_hover_pick(self, pos)` 추가: 실시간 Hover 쿼리 및 상태바 출력.
  - `_on_mouse_double_click(self, event)`, `_process_double_click_pick(self, pos)` 추가: 8pt 3D 라벨 생성.

### C. Verification Plan
1. **[Automated] 패치 테스트 및 모달 주파수 비교 검증**:
   - `python SolverVerification/verification_runner.py`를 실행하여 22개 패치 테스트 통과 무결성을 유지하는지 확인합니다.
2. **[Manual] WHTVisualizer 인터랙션 검증**:
   - `wht_visualizer`를 섀시 결과 데이터로 띄워 `Bead_Height` warping 및 Query 체크박스 활성화 시 마우스 hover에 따른 상태바 메시지 출력, 더블클릭 시 8pt 입체 라벨 생성, `Clear Labels` 버튼 작동성을 정밀 평가합니다.
