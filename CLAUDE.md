# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project: WHT LightChassisModel

FEM 해석 결과를 ParaView로 내보내고, JAX 자동미분 기반으로 경량 Chassis 모델의 토포그래피·두께·탄성계수를 최적화하는 프레임워크.

### Python 실행 환경

conda 환경: **`vdmc`**

`conda run -n vdmc` 는 UTF-8 출력이 포함된 스크립트에서 cp949 인코딩 오류를 발생시키므로, Python 직접 경로를 사용한다:

```bash
# 올바른 실행 방법
"C:/Users/GOODMAN/miniconda3/envs/vdmc/python.exe" script.py

# 단일 명령
"C:/Users/GOODMAN/miniconda3/envs/vdmc/python.exe" -c "import sys; print(sys.version)"
```

멀티라인 테스트는 반드시 `.py` 파일로 작성 후 실행한다. `conda run -c "multiline..."` 형식은 작동하지 않는다.

### 주요 실행 스크립트

```bash
# 기존 modal analysis + ParaView export + PyVista 시각화 (완성된 파이프라인)
cd test_jaxSSO
"C:/Users/GOODMAN/miniconda3/envs/vdmc/python.exe" exam1_nf.py

# wht_converter CLI
"C:/Users/GOODMAN/miniconda3/envs/vdmc/python.exe" -m wht_converter --help
```

---

## 3-패키지 아키텍처

```
wht_converter/   ← 역할 동결. 솔버 결과 → WHTResultData IR → 파일 출력
wht_modeler/     ← FEM 메시 모델 엔티티, LS-DYNA IO
wht_solver/      ← JaxSSO 해석 실행, 최적화, 모니터링
```

**의존 방향** (단방향, 역방향 금지):
```
wht_solver → wht_modeler → wht_converter
```

### wht_converter (동결)

| 클래스 | 역할 |
|--------|------|
| `WHTResultData` | VTK CSR 포맷 FEM 결과 IR. `nodes(N,3)`, `connectivity(K,)`, `offsets(M+1,)`, `cell_types(M,)`, `point_data{name:(T,N,D)}` |
| `WHTMetadata` | 솔버·단위 메타정보 |
| `JaxSSOAdapter` | JaxSSO 결과 → WHTResultData |
| `VTKHDFExporter` | → ParaView `.hdf` (VTKHDF v2.0, `Steps/PartOffsets` 필수) |
| `VTUPVDExporter` | → `.pvd` + `.vtu` |

### wht_modeler

```python
from wht_modeler import WHTMeshModel
from wht_modeler.io import LSDYNAReader, LSDYNAWriter

model = LSDYNAReader().read("chassis.k")
model.apply_spc([0, 1, 2], dofs=(0,1,2,3,4,5))
rd = model.to_wht_result_data()          # → WHTResultData (geometry only)
```

- `WHTMeshModel.nodes`: `dict[int, WHTNode]` (임의 node ID 허용)
- `nodes_array()` → `(N,3)` sorted by nid; `node_id_to_index()` → remapping dict

### wht_solver

```python
from wht_solver import WHTSolver, WHTMapper, LoadCaseLibrary
from wht_solver.wht_optimizer import WHTOptimizer, DesignVariables, DesignBounds
from wht_solver.wht_monitor import OptimizationMonitor

solver = WHTSolver(model)
modal = solver.solve_modal(num_modes=10)     # → WHTSolverResult
static = solver.solve_static(load_case)     # → WHTSolverResult
args = solver.get_k_func_args()             # → K_func static args (for optimizer)
```

**반력 부호 규약**: `result.reaction_force()` 는 JaxSSO Lagrange multiplier `u_aug[ndof:]`를 반환. `sum(R[:,2])` 의 절대값이 총 적용 하중과 같음. 예: `load_z=-1000` → `sum(Rz)=-1000`.

---

## JaxSSO 연동 핵심 사항

**노드 인덱스 규약**: JaxSSO는 0-based contiguous node index를 요구. `nodeTag * 6 = DOF 시작 인덱스`.
WHTSolver 내부에서 `sorted_nids → nid_to_idx` remapping을 자동 처리함.

**K_func 순수 함수** (최적화 루프에서 jit+grad 적용 대상):
```python
from JaxSSO.assemblemodel import K_func
K = K_func(node_crds, ndof, n_beamcol, cnct_beamcols, prop_beamcols,
           n_quad, cnct_quads, prop_quads)
# prop_quads shape: (n_quad, 5) → [t, E, nu, kx_mod, ky_mod]
```

**정적 해석 augmented system**:
```
u_aug = sci_sparse_solve(K_aug, f_aug)
u_free     = u_aug[:ndof]   # 변위
lambda_    = u_aug[ndof:]   # Lagrange multiplier = 반력 (부호 주의)
```

---

## 개발 규칙 및 환경 설정

### 언어 설정
모든 대화와 응답은 **한국어**를 사용한다.

### 인코딩 표준 (Windows UTF-8)
- 모든 `open()` 호출에 `encoding='utf-8'` 명시
- 새 파일은 BOM 없는 UTF-8로 저장
- `UnicodeDecodeError` 발생 시 코드 수정 전 파일 실제 인코딩 확인
- 스크립트 내 Unicode 문자(예: `≈`, `✓`)는 Windows cp949 콘솔에서 오류 발생 → ASCII로 대체

### PyVista 시각화 기본값
- 배경: black, 글자: white (ParaView 스타일)
- Mesh edge color: grey
- 좌표축 표시 기본 on
- 마우스 우클릭 컨텍스트 메뉴: XY/YZ/ZX 투영 + Perspective 전환
- Animation: 객체 재생성 금지, 내부값 update 방식 사용

### 작업 로그
- 구현 계획, 설계 문서, 이슈 → `./dev_log/` 저장, 파일명 `[name]_YYYYMMDD.md`
- 이슈 트래커: `dev_log/issue_tracker.md` — 세션 시작 시 참조해 회귀 방지

---

## Karpathy Guidelines (요약)

코드 작성 시 판단 기준:
1. **구현 전 가정 명시** — 불확실하면 먼저 물어볼 것
2. **요청된 것만 구현** — 추가 추상화, 미래 대비 코드 금지
3. **외과적 수정** — 요청과 무관한 인접 코드 개선·정리 금지
4. **검증 가능한 완료 기준 정의** — "작동하게 만들기"보다 구체적 조건 명시
