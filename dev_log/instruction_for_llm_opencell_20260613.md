# LLM Instruction: 섀시 + 오픈셀(글라스 패널) 결합 해석 및 최적화 기능 개발 지침서

본 문서는 다른 LLM 에이전트가 섀시 단독 최적화 모델에 **오픈셀(글라스 패널) 자동 조립 및 실란트 소프트 빔 연결** 기능을 추가하고, 동적 해석 및 비드 최적화(Topography) 파이프라인에 통합할 수 있도록 상세 작업 사양과 코드 수정 가이드를 제공합니다.

---

## 1. 개발 목표 및 배경 (Context & Goal)
기존의 스틸 섀시 트레이 모델 단독 최적화 파이프라인에서, 디스플레이의 글라스 패널(오픈셀)과 테두리 실란트(연질 체결부)의 물리적 거동을 함께 모사할 수 있도록 어셈블리 모델 생성 알고리즘을 추가합니다.
- **오픈셀 글라스 평판 생성**: 섀시 플랜지 상단에 위치하는 2D 쉘 메시 자동 구성.
- **실란트(소프트 빔) 결합**: 섀시 플랜지와 글라스 패널 가장자리 노드를 소프트 빔(`BEAM2`) 요소로 연결.
- **체결부 연질화 및 하단 제외**: 실란트의 부드러운 거동을 위해 탄성계수 1.0 MPa를 부여하며, 전면 하단부(Y-min)는 체결에서 제외.
- **최적화 설계 변수 필터링**: 최적화 과정에서 글라스 및 실란트 요소가 비드 변형 대상에서 제외되도록 처리.

---

## 2. 주요 대상 파일 및 작업 상세

### 2.1 [MODIFY] `wht_topo/run_topo.py`
오픈셀 글라스 패널을 자동 생성하고 섀시와 조립하는 `_build_chassis_with_glass_assembly` 함수를 탑재하고, 관련 CLI 옵션을 추가합니다.

#### A. CLI Argument Parser 추가
```python
# argparse 또는 gooey parser 설정 영역에 다음 옵션을 추가합니다.
g2.add_argument("--add-glass", action="store_true", default=False,
                help="오픈셀 글라스 패널을 섀시 모델에 자동으로 결합하여 조립 모델 구성")
g2.add_argument("--glass-t", type=float, default=1.0,
                help="글라스 패널 두께 mm (기본: 1.0)")
g2.add_argument("--glass-E", type=float, default=40000.0,
                help="글라스 패널 탄성 계수 MPa (기본: 40000.0)")
g2.add_argument("--sealant-E", type=float, default=1.0,
                help="소프트 결합 실란트 빔의 탄성 계수 MPa (기본: 1.0)")
```

#### B. 어셈블리 생성 알고리즘 구현 (`_build_chassis_with_glass_assembly`)
1. **플랜지 최종단 검색**: 섀시 쉘 노드 중 최대 Z 좌표($Z_{max}$)를 찾아 플랜지 단 노드를 식별합니다.
2. **글라스 생성 크기 및 위치 결정**:
   - 플랜지 단 노드들의 X, Y 좌표 범위를 기반으로 Bounding Box ($[X_{min}, X_{max}], [Y_{min}, Y_{max}]$)를 구합니다.
   - 글라스 평판은 Z축 방향으로 시각적 식별성을 높이기 위해 $Z_{glass} = Z_{max} + 10.0$ mm 평면에 위치시킵니다.
3. **글라스 평판 메시 생성**:
   - Bounding Box 영역에 섀시와 유사한 격자 크기(`mesh_xy`)를 반영해 `QUAD4` 쉘 요소를 생성합니다.
   - 글라스 패널 속성: Property ID `pid = 3`, Material ID `mid = 3` (탄성계수 $40,000$ MPa, 밀도 $2.5 \times 10^{-9}$ tonne/mm³).
4. **실란트 소프트 빔 (`BEAM2`) 연결 및 Y-min 제외**:
   - 글라스 평판의 외곽 가장자리 노드 중 **전면 하단(Y-min) 라인을 원천 배제**합니다.
     - 조건식: `on_boundary = (abs(x - x_min) < 0.1 or abs(x - x_max) < 0.1 or abs(y - y_max) < 0.1)` (즉, `y - y_min` 조건은 필터링에서 제외)
   - 남은 가장자리 노드와 섀시 플랜지 단 노드 간에 XY 평면 기준 유클리드 거리가 가장 가까운 쌍을 1:1로 매핑합니다.
   - 매핑된 두 노드 사이를 `BEAM2` 요소로 결합합니다. (거리 임계값: `1.5 * mesh_xy`)
   - 실란트 빔 속성: Property ID `pid = 2`, Material ID `mid = 2` (탄성계수 `sealant_E` (기본 1.0 MPa), 밀도 거의 0인 $1.0 \times 10^{-15}$ tonne/mm³).

---

### 2.2 [MODIFY] `wht_topo/solver.py`
최적화 진행 도중 글라스 패널과 실란트 빔의 형상이 찌그러지거나 강제로 비드 설계 변수에 들어가지 않도록 설계 변수 탐색에서 제외합니다.

- **위치**: `WHTopographySolver._find_design_elements(self)`
- **수정 사양**:
  - 모델의 모든 요소를 순회하면서 설계 변수 후보를 선택할 때, 요소의 Property ID(`pid`)가 **`2` (소프트 빔) 또는 `3` (글라스 패널) 인 경우 대상에서 제외**합니다.
```python
for eid in self.elem_ids:
    elem = self.model.elements[eid]
    if elem.pid in (2, 3):  # 글라스 패널 및 실란트 빔 제외
        continue
    # ... 기존 설계 변수 판단 및 1-ring 침식 로직 ...
```

---

### 2.3 [MODIFY] `wht_modeler/wht_dynamic_utils.py`
Windows PowerShell 및 Matplotlib 기반의 시각화 파이프라인에서 한국어 인코딩(`cp949`) 오류가 발생하지 않도록 전처리 코드 및 텍스트 출력을 점검합니다.
- 파일 쓰기 및 읽기(`open`) 발생 시 명시적으로 `encoding='utf-8'` 옵션을 사용해야 합니다.
- 텍스트 혹은 그래프 제목 등에 Em dash(`—`, `\u2014`) 등 CP949로 인코딩이 불가능한 특수 문자가 존재할 경우 일반 아스키 대시(`-`)로 치환하여 `UnicodeEncodeError`를 사전 방지해야 합니다.

---

### 2.4 [NEW] `test_jaxSSO/exam5_dynamic_with_oc.py`
섀시 단독 동적 해석 및 오픈셀 조립 기능만을 고립시켜 빠르게 테스트할 수 있는 검증 예제 스크립트를 작성합니다.
- `_build_chassis_with_glass_assembly` 함수를 통해 조립을 구동하고,
- 실측 코너 가속도 CSV 데이터를 기반으로 Kabsch 전처리 및 관성 하중 인가 후,
- `solve_modal` 및 과도 동역학 해석이 수치적 특이성(Singularity) 없이 정상 구동하여 결과를 HDF 형식으로 저장하는 전체 워크플로우를 테스트하도록 설계합니다.

---

## 3. 검증 기준 (Success Criteria)

수정을 완료한 후, 반드시 터미널 명령을 통해 수치 및 수렴 안정성을 확인해야 합니다.

### 3.1 단독 해석 검증
```bash
# 가상환경의 python.exe를 직접 사용하며, 인코딩 에러 방지를 위해 PowerShell 출력 인코딩을 UTF-8로 설정합니다.
$OutputEncoding = [System.Text.UTF8Encoding]::new(); [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; python test_jaxSSO/exam5_dynamic_with_oc.py --no-viz
```
- **합격 기준**:
  - 소프트 연결 빔(`BEAM2`) 개수가 **270개**로 구성되어야 함 (Y-min 라인이 정상 제외된 경우).
  - 고유치 해석(NF)에서 1차 탄성 모드 주파수가 **1.26 Hz** 내외로 수렴해야 함 (체결부 강성이 1.0 MPa로 완화된 연질 거동 확인).
  - 모달 과도 해석이 수치적 폭주(NaN 또는 무한대 변위) 없이 정상 수렴하고 파일로 출력되어야 함.

### 3.2 최적화 파이프라인 검증
```bash
python wht_topo/run_topo.py --iters 3 --add-glass --n-modes 20 --dynamic-opts wht_topo/structural_dynamics_c235.csv --add-inertia --ignore-gooey --no-viz --no-gui
```
- **합격 기준**:
  - 조립된 모델의 NDOF가 40,000 자유도를 상회하더라도 에러 없이 3 iterations 최적화가 정상 진행되어야 함.
  - 최적화 루프 내에서 고유치 주파수가 1.1 Hz 대역을 형성하며 목적함수가 수렴해야 함.
  - 최종 최적 비드가 적용된 `industrial_bead.k` 파일이 안전하게 생성되어야 함.
