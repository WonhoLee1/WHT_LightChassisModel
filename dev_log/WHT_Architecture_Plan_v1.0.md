# WHT FEM Framework — Architecture Plan v1.0

**Date**: 2026-04-18
**Status**: Plan (Pending Implementation)
**Scope**: 3-패키지 아키텍처 설계 + 구현 로드맵

---

## 0. 배경 및 목표

**최종 목표**: 복잡한 Chassis 구조(High-fidelity)의 구조적 거동을,
단순화된 트레이/평판 모델(Low-fidelity)이 최대한 모사하도록
토포그래피·두께·탄성계수를 JAX 자동미분 기반으로 최적화한다.

**현재 상태**: `wht_converter` (IR + Adapter + Exporter) 구현 완료.

---

## 1. 3-패키지 아키텍처

```
WHT_LightChassisModel/
│
├── wht_converter/      ← 현재 구현됨. 역할 동결.
│   역할: 솔버 결과 → WHTResultData IR → 파일 출력
│
├── wht_modeler/        ← NEW (Phase 1~2)
│   역할: 메시 모델 엔티티 관리, FEM 파일 읽기/쓰기, RBE2
│
└── wht_solver/         ← NEW (Phase 3~5)
    역할: 해석 실행, 반력 계산, 매핑, 최적화 루프
```

### 패키지 간 의존 방향

```
wht_solver
    │  uses
    ├──▶ wht_modeler   (WHTMeshModel, RBE2, LoadCase)
    └──▶ wht_converter (WHTResultData → 파일 출력)

wht_modeler
    │  uses
    └──▶ wht_converter (WHTResultData geometry fields)

[의존성 규칙] converter ← modeler ← solver (단방향, 역방향 금지)
```

---

## 2. `wht_converter` — 역할 동결

현재 구현 그대로 유지. 추가 기능 없음.

| 클래스 | 역할 |
|--------|------|
| `WHTMetadata` | 솔버·단위·분석 유형 메타정보 |
| `WHTResultData` | 결과 데이터 IR (nodes, connectivity, point_data, ...) |
| `JaxSSOAdapter` | JaxSSO 결과 → WHTResultData |
| `JaxFEMAdapter` | jax-fem 결과 → WHTResultData |
| `VTKHDFExporter` | → ParaView `.hdf` |
| `VTUPVDExporter` | → `.pvd` + `.vtu` |
| `HWASCIIExporter` | → Altair HyperView `.ascii` |

---

## 3. `wht_modeler` — 상세 설계

### 3.1 파일 구조

```
wht_modeler/
├── __init__.py
├── wht_mesh_model.py     ← WHTMeshModel (핵심 엔티티)
├── wht_entities.py       ← WHTNode, WHTElement, WHTSet, WHTRBE2, WHTProperty, WHTMaterial
├── io/
│   ├── __init__.py
│   ├── base_reader.py    ← BaseFEMReader (ABC)
│   ├── base_writer.py    ← BaseFEMWriter (ABC)
│   ├── lsdyna_reader.py  ← LSDYNAReader
│   ├── lsdyna_writer.py  ← LSDYNAWriter
│   ├── optistruct_reader.py
│   ├── optistruct_writer.py
│   ├── radioss_reader.py
│   ├── radioss_writer.py
│   ├── abaqus_reader.py
│   └── abaqus_writer.py
└── tests/
    ├── test_mesh_model.py
    ├── test_lsdyna_io.py
    └── test_other_io.py
```

---

### 3.2 `WHTMeshModel` — 핵심 엔티티

```python
class WHTMeshModel:
    """
    FEM 전처리 데이터 컨테이너.
    IO 리더가 파싱한 결과를 저장하고, 사용자가 BC/하중을 설정하는 인터페이스를 제공한다.
    """

    # ── Geometry ──────────────────────────────────────────
    nodes:    dict[int, WHTNode]       # {nid: WHTNode(x,y,z)}
    elements: dict[int, WHTElement]    # {eid: WHTElement}

    # ── Sets ──────────────────────────────────────────────
    node_sets: dict[int, WHTNodeSet]   # {set_id: WHTNodeSet}
    elem_sets: dict[int, WHTElemSet]   # {set_id: WHTElemSet}

    # ── RBE2 ──────────────────────────────────────────────
    rbe2s: dict[int, WHTRBE2]          # {rbe2_id: WHTRBE2(master, slaves)}

    # ── Properties / Materials ────────────────────────────
    properties: dict[int, WHTProperty] # {pid: WHTProperty(t, E, nu, rho)}
    materials:  dict[int, WHTMaterial] # {mid: WHTMaterial(E, nu, rho)}

    # ── BCs & Loads (사용자가 코드에서 설정) ───────────────
    spc_conditions: list[WHTSPCEntry]  # [(node_ids, dofs, value)]
    loads:          list[WHTLoadEntry] # [(node_ids, dofs, values)]
```

**핵심 API**:

```python
# Set 기반 노드/요소 조회
model.get_nodes_by_set(set_id: int) -> list[int]
model.get_elems_by_set(set_id: int) -> list[int]

# RBE2 기반 노드 조회
model.get_rbe2_slaves(master_nid: int) -> list[int]
model.get_rbe2_masters() -> list[int]

# BC / Load 설정 (set_id 또는 node_id list 모두 허용)
model.apply_spc(target: int | list[int], dofs=(0,1,2,3,4,5), value=0.0)
model.apply_force(target: int | list[int], dofs=(0,1,2), values=(0,0,-1000))

# wht_converter IR로 변환 (geometry만, 결과 없음)
model.to_wht_result_data() -> WHTResultData
```

---

### 3.3 엔티티 클래스

```python
@dataclass
class WHTNode:
    nid: int
    x: float; y: float; z: float

@dataclass
class WHTElement:
    eid: int
    type: str              # "QUAD4" | "TRIA3" | "BEAM2" | "SOLID8" | ...
    node_ids: list[int]
    pid: int               # property ID

@dataclass
class WHTNodeSet:
    sid: int
    node_ids: list[int]    # LIST / BOX / PART 파싱 결과 통합 보관

@dataclass
class WHTElemSet:
    sid: int
    elem_ids: list[int]

@dataclass
class WHTRBE2:
    rbe2_id: int
    master_nid: int
    slave_nids: list[int]
    dofs: tuple            # 구속 자유도 (기본: (0,1,2,3,4,5))

@dataclass
class WHTProperty:
    pid: int
    type: str              # "PSHELL" | "PBEAM" | ...
    t: float               # shell 두께
    mid: int               # material ID 참조

@dataclass
class WHTMaterial:
    mid: int
    E: float; nu: float; rho: float
```

---

### 3.4 IO 리더 — 지원 키워드

#### LSDYNAReader (`.k` / `.key`)

| 키워드 | 지원 내용 |
|--------|-----------|
| `*NODE` | nid, x, y, z |
| `*ELEMENT_SHELL` | eid, pid, n1~n4 (QUAD4) |
| `*ELEMENT_BEAM` | eid, pid, n1, n2 |
| `*PART` | pid, mid, title |
| `*MAT_*` | E, nu, rho (주요 재료 카드) |
| `*SECTION_SHELL` | t (두께) |
| `*SET_NODE_LIST` | ID 리스트 (고정폭 8열) |
| `*SET_NODE_GENERAL` | **BOX** (xmin~zmax 범위), **PART** (pid 기준) |
| `*SET_ELEMENT_LIST` | ID 리스트 |
| `*SET_ELEMENT_GENERAL` | BOX, PART |
| `*CONSTRAINED_NODAL_RIGID_BODY` | RBE2 동등 (master + slave list) |

#### OptistructReader (`.bdf` / `.fem`)

| 키워드 | 지원 내용 |
|--------|-----------|
| `GRID` | nid, x, y, z |
| `CQUAD4` | eid, pid, n1~n4 |
| `PSHELL` | pid, mid, t |
| `MAT1` | mid, E, nu, rho |
| `SET` | id, list (연속/범위 형식) |
| `SPC` / `SPC1` | BC |
| `FORCE` | 집중 하중 |
| `RBE2` | master, dofs, slaves |

#### RadiossReader (`.rad` / `_0000.rad`)

| 키워드 | 지원 내용 |
|--------|-----------|
| `/NODE` | nid, x, y, z |
| `/SHELL` | eid, pid, nids |
| `/PROP/SHELL` | t |
| `/MAT/LAW1` | E, nu, rho |
| `/SET/GENERAL` | 노드/요소 집합 |
| `/RBE2` (또는 `/RBODY`) | rigid body |

#### AbaqusReader (`.inp`)

| 키워드 | 지원 내용 |
|--------|-----------|
| `*NODE` | nid, x, y, z |
| `*ELEMENT, TYPE=S4R` | quad shell |
| `*SHELL SECTION` | t, material |
| `*MATERIAL` | E, nu, density |
| `*NSET, NSET=name` | node set (list 형식) |
| `*ELSET, ELSET=name` | element set |
| `*RIGID BODY` | RBE2 동등 |

#### BaseFEMWriter — 공통 출력

```python
class BaseFEMWriter(ABC):
    def write(self, model: WHTMeshModel, path: str) -> None: ...

# 구현 대상
LSDYNAWriter     → .k  (가장 먼저 구현)
OptistructWriter → .bdf
RadiossWriter    → .rad
AbaqusWriter     → .inp
```

---

### 3.5 RBE2 → Stiff Beam 변환

`wht_solver`에서 JaxSSO 모델을 빌드할 때 `WHTRBE2`를 stiff beam으로 변환.

```python
# wht_solver 내부에서 사용
def rbe2_to_stiff_beams(
    rbe2: WHTRBE2,
    k_max: float,             # 시스템 강성 최대값
    stiffness_scale: float = 1e3,   # 기본값 10^3 (사용자 설정 가능)
) -> list[BeamEntry]:
    """
    master → 각 slave 노드를 stiff beam으로 연결.
    E_rbe = stiffness_scale × k_max
    A = 1.0 (단위 면적), I = 매우 큰 값

    수치 안정성 고려:
    - stiffness_scale이 너무 크면 condition number 악화
    - 기본 10^3은 구조 강성 대비 충분히 강하지만 cond(K) < 10^12 유지
    - 필요시 10^6까지 허용 (경고 발생)
    """
```

---

## 4. `wht_solver` — 상세 설계

### 4.1 파일 구조

```
wht_solver/
├── __init__.py
├── wht_solver.py          ← WHTSolver (JaxSSO wrapper)
├── wht_result.py          ← WHTSolverResult (반력 API 포함)
├── wht_mapper.py          ← WHTMapper (RBF 보간)
├── wht_optimizer.py       ← WHTOptimizer (JAX + Optax)
├── load_cases.py          ← LoadCaseLibrary (프리셋)
├── objectives.py          ← MAC, 주파수차이, RMSE 목적함수
└── tests/
    ├── test_solver.py
    ├── test_mapper.py
    └── test_optimizer.py
```

---

### 4.2 `WHTSolver` — JaxSSO 래퍼

```python
class WHTSolver:
    """
    WHTMeshModel을 받아 JaxSSO 모델을 빌드하고 해석을 실행한다.
    RBE2를 stiff beam으로 자동 변환한다.
    """

    def __init__(self, model: WHTMeshModel, stiffness_scale: float = 1e3):
        self.model = model
        self.stiffness_scale = stiffness_scale

    def solve_static(self, load_case: WHTLoadCase) -> WHTSolverResult:
        """정적 해석. BCs + 하중을 JaxSSO에 적용 후 해석."""

    def solve_modal(self, num_modes: int = 10) -> WHTSolverResult:
        """모달 해석. 고유진동수 + 고유벡터 반환."""

    def solve_all(
        self,
        load_cases: list[WHTLoadCase],
        num_modes: int = 10,
    ) -> dict[str, WHTSolverResult]:
        """전체 하중 케이스 + 모달 일괄 실행."""
```

---

### 4.3 `WHTSolverResult` — 결과 및 반력 API

```python
class WHTSolverResult:
    """
    해석 결과 컨테이너. 최적화 목적함수·제약함수에 직접 사용 가능.
    """
    analysis_type: str          # "static" | "modal"
    node_ids: list[int]         # 정렬된 노드 ID 목록

    # 변위
    displacement: np.ndarray    # (N, 6) — 6 DOF per node [ux,uy,uz,rx,ry,rz]

    # 모달 (analysis_type="modal"일 때만)
    frequencies: np.ndarray     # (n_modes,) [Hz]
    mode_shapes: np.ndarray     # (n_modes, N, 6)

    # 반력 (정적 해석에서만 의미 있음)
    # 내부 계산: R = K @ u_full - F_ext
    # R[i*6:i*6+6] = i번째 노드의 반력 벡터
    _reaction_force_full: np.ndarray  # (N*6,) internal

    def reaction_force(
        self,
        node_ids: int | list[int] | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        node_ids : None → 전체 노드 (N, 3) 반환
                   int  → 해당 노드 (3,) 반환 [Rx, Ry, Rz]
                   list → 선택 노드 (len, 3) 반환

        Returns
        -------
        np.ndarray — 병진 반력 [Rx, Ry, Rz]만 반환 (회전 반력 제외)

        Note
        ----
        최적화 제약함수 예시:
            r = result.reaction_force(support_set_nodes)
            total_reaction = jnp.sum(r[:, 2])   # Z 방향 합력
        """

    def to_wht_result_data(self, metadata: WHTMetadata) -> WHTResultData:
        """wht_converter 포맷으로 변환 → ParaView 출력용."""
```

---

### 4.4 `WHTMapper` — RBF 보간 (Hi-fi → Lo-fi 매핑)

```python
class WHTMapper:
    """
    서로 다른 메시 밀도의 두 모델 간 결과 데이터를 보간한다.
    타겟(Hi-fi) 결과를 최적화 모델(Lo-fi) 노드 위치로 매핑.

    구현: scipy.interpolate.RBFInterpolator (thin-plate spline 기본)
    타겟 변경 지원: re-fit 구조 제공 (구조적 큰 변화 없이 update 가능)
    """

    def fit(
        self,
        source_nodes: np.ndarray,   # (N_hi, 3) Hi-fi 노드 좌표
        source_data:  np.ndarray,   # (N_hi, D) 매핑할 데이터 (변위, 모드벡터 등)
        kernel: str = "thin_plate_spline",
    ) -> None:
        """RBF 인터폴레이터 구성. 타겟 변경 시 re-fit 호출."""

    def transform(
        self,
        target_nodes: np.ndarray,   # (N_lo, 3) Lo-fi 노드 좌표
    ) -> np.ndarray:                # (N_lo, D) 매핑된 데이터
        """Lo-fi 노드 위치로 Hi-fi 데이터 보간."""

    def fit_transform(self, source_nodes, source_data, target_nodes):
        """편의 메서드."""
```

---

### 4.5 `LoadCaseLibrary` — 하중 케이스 프리셋

```python
class LoadCaseLibrary:
    """
    표준 구조 하중 케이스를 WHTMeshModel에 한 줄로 적용.
    set_id 또는 node_id list 모두 허용.
    """

    @staticmethod
    def three_point_bending(
        model: WHTMeshModel,
        support_sets: list[int | list[int]],  # 2개 지지점
        load_target: int | list[int],          # 중앙 가력점
        load_z: float = -1000.0,               # [N]
        constrain_dofs: tuple = (0,1,2),
    ) -> WHTLoadCase:

    @staticmethod
    def four_point_bending(
        model: WHTMeshModel,
        support_sets: list[int | list[int]],  # 2개 지지점
        load_targets: list[int | list[int]],  # 2개 가력점
        load_z: float = -1000.0,
    ) -> WHTLoadCase:

    @staticmethod
    def twisting(
        model: WHTMeshModel,
        fixed_corner: int | list[int],
        twist_corner: int | list[int],
        load_z: float = -1000.0,
    ) -> WHTLoadCase:

    @staticmethod
    def corner_lift(
        model: WHTMeshModel,
        support_sets: list[int | list[int]],  # 3개 코너 고정
        lift_target: int | list[int],          # 1개 코너 가력
        load_z: float = 1000.0,
    ) -> WHTLoadCase:

    @staticmethod
    def end_bending(
        model: WHTMeshModel,
        fixed_end: int | list[int],
        load_end: int | list[int],
        load_z: float = -1000.0,
    ) -> WHTLoadCase:
```

---

### 4.6 `objectives.py` — 목적함수 (JAX 기반)

```python
# ── MAC ──────────────────────────────────────────────────────────────────
def mac(phi_a: jnp.ndarray, phi_b: jnp.ndarray) -> jnp.ndarray:
    """
    표준 MAC (벡터 내적 기반).

    MAC(a, b) = (φ_a · φ_b)^2 / ((φ_a · φ_a)(φ_b · φ_b))

    전제: phi_a, phi_b는 같은 길이 (RBF 매핑 후)
    반환: scalar ∈ [0, 1].  1.0 = 완전 일치
    """

def mass_weighted_mac(
    phi_a: jnp.ndarray,   # (N,) 또는 (N, 3) 매핑된 모드벡터
    phi_b: jnp.ndarray,
    M_diag: jnp.ndarray,  # (N,) lumped mass 대각 성분
) -> jnp.ndarray:
    """
    질량 가중 MAC.

    MAC_M(a,b) = (φ_a^T M φ_b)^2 / ((φ_a^T M φ_a)(φ_b^T M φ_b))

    용도: 서로 다른 메시 밀도에서 에너지 분포 유사도 비교.
    RBF 매핑 후 사용. 저해상도 모델의 lumped M 사용.
    """

# ── Multi-objective loss ──────────────────────────────────────────────────
def multi_objective_loss(
    opt_result:    WHTSolverResult,
    target_result: WHTSolverResult,
    mapper:        WHTMapper,
    weights: dict = {"freq": 1.0, "mac": 1.0, "static": 1.0},
    use_mass_mac:  bool = False,
) -> jnp.ndarray:
    """
    f_obj = w1 * Δfreq_loss
          + w2 * (1 - MAC)_loss
          + w3 * RMSE(disp)_loss

    Δfreq_loss: 각 모드 주파수 차이의 MSE
    MAC_loss:   모든 모드에 대해 (1 - MAC) 합산
    RMSE_loss:  각 하중 케이스 변위 RMSE 합산
    """
```

---

### 4.7 `WHTOptimizer` — 최적화 엔진

```python
@dataclass
class DesignVariables:
    """
    JAX pytree로 등록되어 jax.grad / optax 직접 사용 가능.
    """
    t_field:   jnp.ndarray   # (M,) 요소별 두께 [mm]
    z_offsets: jnp.ndarray   # (N,) 노드별 Z-offset (토포그래피) [mm]
    E:         float          # 전체 탄성계수 [MPa]
    rho:       float          # 전체 밀도 [t/mm³]

@dataclass
class DesignBounds:
    t_min: float;   t_max: float
    z_min: float;   z_max: float   # 토포그래피 최대 높이
    E_min: float;   E_max: float
    rho_min: float; rho_max: float

class WHTOptimizer:
    """
    Optax 기반 gradient descent 최적화.
    JAX jit + vmap으로 복수 하중 케이스를 병렬 처리.
    """

    def __init__(
        self,
        base_model: WHTMeshModel,
        target_results: dict[str, WHTSolverResult],
        mapper: WHTMapper,
        bounds: DesignBounds,
        load_cases: list[WHTLoadCase],
        num_modes: int = 10,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        weights: dict = {"freq": 1.0, "mac": 1.0, "static": 1.0},
    ):

    def run(
        self,
        init_vars: DesignVariables,
        n_steps: int = 500,
        log_every: int = 10,
    ) -> tuple[DesignVariables, list[float]]:
        """
        최적화 루프.
        Returns: (최적 설계변수, loss 히스토리)
        """
```

---

## 5. 전체 워크플로우 (End-to-End)

```python
# ── Step 1. 타겟 모델 로드 및 해석 ─────────────────────────────────────
target_model = LSDYNAReader().read("chassis_target.k")
target_solver = WHTSolver(target_model)

target_results = {
    "modal":         target_solver.solve_modal(num_modes=10),
    "3pt_bending":   target_solver.solve_static(LoadCaseLibrary.three_point_bending(...)),
    "4pt_bending":   target_solver.solve_static(LoadCaseLibrary.four_point_bending(...)),
    "corner_lift":   target_solver.solve_static(LoadCaseLibrary.corner_lift(...)),
    "twisting":      target_solver.solve_static(LoadCaseLibrary.twisting(...)),
    "end_bending":   target_solver.solve_static(LoadCaseLibrary.end_bending(...)),
}

# ── Step 2. 최적화 모델 생성 (단순 트레이, 메시 3배 크기) ────────────────
from mesh_utils import generate_shell_tray
opt_model = WHTMeshModel.from_node_elem_db(*generate_shell_tray(...))

# ── Step 3. 노드 매핑 준비 ───────────────────────────────────────────────
mapper = WHTMapper()
# 각 하중 케이스별로 타겟 변위를 opt_model 노드로 매핑
mapper.fit(
    source_nodes = target_model.nodes_array(),
    source_data  = target_results["modal"].mode_shapes[0],  # 1st mode
)

# ── Step 4. 최적화 실행 ─────────────────────────────────────────────────
bounds = DesignBounds(
    t_min=0.5, t_max=5.0,
    z_min=0.0, z_max=10.0,   # 토포그래피 최대 높이
    E_min=500, E_max=5000,
    rho_min=5e-10, rho_max=5e-9,
)
init_vars = DesignVariables(
    t_field   = jnp.full(opt_model.n_elements, 2.0),
    z_offsets = jnp.zeros(opt_model.n_nodes),
    E         = 1000.0,
    rho       = 1e-9,
)
optimizer = WHTOptimizer(
    base_model     = opt_model,
    target_results = target_results,
    mapper         = mapper,
    bounds         = bounds,
    load_cases     = [...],
    weights        = {"freq": 2.0, "mac": 1.0, "static": 1.0},
)
best_vars, loss_history = optimizer.run(init_vars, n_steps=500)

# ── Step 5. 결과 출력 ───────────────────────────────────────────────────
final_solver = WHTSolver(opt_model.apply_design(best_vars))
final_result = final_solver.solve_modal()
final_result.to_wht_result_data(meta).export("results/optimized.hdf")
```

---

## 6. 구현 로드맵

### Phase 1 — `wht_modeler` Core (wht_mesh_model + entities)

- [ ] `wht_entities.py`: WHTNode, WHTElement, WHTSet, WHTRBE2, WHTProperty, WHTMaterial
- [ ] `wht_mesh_model.py`: WHTMeshModel (CRUD API + apply_spc/apply_force)
- [ ] `tests/test_mesh_model.py`

**완료 기준**: `WHTMeshModel`을 직접 dict에서 구성하고 BC/Load 설정 후 JaxSSO 모델로 변환 가능

---

### Phase 2 — LS-DYNA IO (Reader + Writer 우선)

- [ ] `io/lsdyna_reader.py`: *NODE, *ELEMENT_SHELL, *SET_NODE_LIST/GENERAL(BOX, PART), RBE2
- [ ] `io/lsdyna_writer.py`: WHTMeshModel → .k 파일
- [ ] `tests/test_lsdyna_io.py`: 왕복 변환 테스트 (read → write → re-read 일치)

**완료 기준**: `exam1_nf.py`의 트레이 메시를 .k로 저장 후 재로드하여 해석 실행

---

### Phase 3 — `wht_solver` Core (Solver + Result + ReactionForce)

- [ ] `wht_solver.py`: WHTSolver (JaxSSO wrapper + RBE2→stiff beam)
- [ ] `wht_result.py`: WHTSolverResult + reaction_force() API
- [ ] `load_cases.py`: LoadCaseLibrary (3pt/4pt bending, twisting, corner_lift, end_bending)
- [ ] `tests/test_solver.py`

**완료 기준**: WHTMeshModel로 modal + 복수 정적 해석 실행, 반력 값 코드에서 접근 가능

---

### Phase 4 — Mapper + MAC

- [ ] `wht_mapper.py`: WHTMapper (RBF, re-fit 지원)
- [ ] `objectives.py`: mac(), mass_weighted_mac(), multi_objective_loss()
- [ ] `tests/test_mapper.py`

**완료 기준**: 고해상도 타겟 모드벡터를 저해상도 노드로 매핑, MAC 값 계산

---

### Phase 5 — Optimizer + 나머지 IO

- [ ] `wht_optimizer.py`: DesignVariables, DesignBounds, WHTOptimizer (Optax Adam)
- [ ] `io/optistruct_reader.py`, `io/radioss_reader.py`, `io/abaqus_reader.py`
- [ ] `tests/test_optimizer.py`

**완료 기준**: 단순 트레이가 chassis 타겟과 유사한 고유진동수 + MAC를 내도록 최적화 수렴

---

## 7. 핵심 기술 결정 (확정)

| 항목 | 결정 | 근거 |
|------|------|------|
| RBE2 구현 | Stiff Beam (stiffness_scale 옵션) | JaxSSO 수정 불필요, JAX autodiff 안정 |
| RBE2 기본 강성비 | `1e3` (10^3) | 구조 강성 대비 충분, cond(K) < 10^12 유지 |
| 반력 계산 | `R = K·u - F_ext` | K, u는 해석 후 접근 가능 |
| 반력 API | 전체 ndarray (N,3), node_id 슬라이싱 | 최적화 제약함수와 직접 연동 |
| MAC 기본 | 표준 벡터 내적 MAC (RBF 매핑 후) | 단순, JAX jit 호환 |
| MAC 옵션 | `mass_weighted_mac()` 별도 제공 | 에너지 기반 비교가 필요할 때 |
| 최적화 | JAX + Optax (Adam 기본) | JaxSSO 자동미분 생태계 일치 |
| 보간법 | scipy RBFInterpolator (thin-plate spline) | 전처리 1회, 루프 중 재사용 |
| SET 지원 | LIST + GENERAL(BOX, PART) | 사용자 요구사항 |

---

*이 계획은 구현 시작 전 검토·승인 대상입니다.*
