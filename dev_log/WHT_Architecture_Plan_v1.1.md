# WHT FEM Framework — Architecture Plan v1.1

**Date**: 2026-04-18
**Status**: Plan (Pending Implementation)
**변경 요약**: v1.0 대비 8개 보충 항목 반영 + JaxSSO 토포그래피 갭 분석 + PyVistaQt 모니터링 + 반력 전략 확정

---

## 변경 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|-----------|
| v1.0 | 2026-04-18 | 3-패키지 아키텍처 초안 |
| v1.1 | 2026-04-18 | JAX jit 경계 전략 확정, 반력 전략 변경(Lagrange→), 모달 자동미분 설계, 모드 스위칭 해결, 위상 스모크 테스트 추가, PyVistaQt 모니터링 설계, JaxSSO 토포그래피 갭 분석 |

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
    역할: 해석 실행, 반력 계산, 매핑, 최적화 루프, 진행 모니터링
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

## 2A. JaxSSO 토포그래피 기능 갭 분석 [NEW v1.1]

### JaxSSO가 제공하는 것 (`SSO_model`)

JaxSSO는 `SSO_model` 클래스를 통해 기본 토포그래피 최적화 기능을 제공한다.

```python
# JaxSSO SSO_model 주요 API
sso = SSO_model(jax_model)

# 노드 Z좌표를 설계변수로 등록
sso.NodeParameter(nid, parameter_type=2)  # 2 = Z-coord

# 요소 물성을 설계변수로 등록
sso.ElementParameter(eid, parameter_type=0)  # 0=t, 1=E, 2=nu

# @jit로 데코레이트된 파라미터 업데이트 함수
params = sso.params_u()   # 현재 설계변수 배열 반환

# 순수 JAX 함수 — jit+grad 가능 (핵심)
K = K_func(node_crds, ndof, n_bc, cnct_bc, prop_bc, n_quad, cnct_quads, prop_quads)

# 목적함수 + 그래디언트 (strain energy 또는 사용자 정의)
val, grad = sso.value_and_grad(params)
```

| 제공 기능 | 상세 |
|-----------|------|
| 노드 Z-coord 최적화 | `NodeParameter(nid, 2)` |
| 요소 두께/탄성계수/포아송비 최적화 | `ElementParameter(eid, 0/1/2)` |
| JAX jit/grad 호환 | `K_func`가 pure function (Python side-effect 없음) |
| 정적 해석 + adjoint | `custom_vjp` wrapper in `solver.py` |
| GD + SLSQP | `Optimization` class |

### JaxSSO에서 부족한 것 (→ wht_solver가 보완)

| 부족한 기능 | 현황 | wht_solver 구현 방향 |
|-------------|------|----------------------|
| 다목적 손실 함수 | strain energy 또는 user 단일 함수만 | `multi_objective_loss()` (freq + MAC + RMSE) |
| 모달 최적화 | 지원 없음 (`eigsh`는 autodiff 불가) | `jnp.linalg.eigh` + 고유값 감도식 |
| MAC 계산 | 없음 | `objectives.py`: `mac()`, `mass_weighted_mac()` |
| 크로스 메시 보간 | 없음 | `WHTMapper` (RBF thin-plate spline) |
| 하중 케이스 라이브러리 | 없음 | `LoadCaseLibrary` (6종 프리셋) |
| RBE2 지원 | 없음 | stiff beam 변환 |
| FEM 파일 I/O | 없음 | `wht_modeler` IO 리더/라이터 |
| Optax 통합 | GD vanilla + SLSQP (nlopt) only | Optax Adam + 구속 프로젝션 |
| GD bounds | 없음 | `jnp.clip()` 프로젝션 |
| 부드러움 정규화 | 없음 | Laplacian regularization |
| 모드 순서 추적 | 없음 | MAC 행렬 기반 soft assignment |
| 최적화 모니터링 | print 출력만 | PyVistaQt BackgroundPlotter |

### 전략 결정

```
JaxSSO SSO_model 직접 사용 ✗
→ 이유: SSO_model의 value_and_grad는 JaxSSO Python 객체에 종속.
         Multi-step 최적화에서 Python 객체를 매 스텝 재빌드하는 비용 발생.

채택 전략 (Strategy B): K_func 직접 호출
→ assemblemodel.K_func(node_crds, ndof, n_bc, cnct_bc, prop_bc,
                        n_quad, cnct_quads, prop_quads) 는 pure JAX function
→ JaxSSO Python Model 객체 없이 jit+grad 적용 가능
→ DesignVariables pytree (t_field, z_offsets, E, rho) → node_crds, prop_quads 변환 후
  K_func에 직접 전달
```

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

    수치 안정성:
    - stiffness_scale이 너무 크면 condition number 악화
    - 기본 10^3은 cond(K) < 10^12 유지
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
├── wht_monitor.py         ← OptimizationMonitor (PyVistaQt) [NEW v1.1]
├── load_cases.py          ← LoadCaseLibrary (프리셋)
├── objectives.py          ← MAC, 주파수차이, RMSE 목적함수
└── tests/
    ├── test_solver.py
    ├── test_mapper.py
    └── test_optimizer.py
```

---

### 4.2 JAX jit 경계 전략 [NEW v1.1]

**핵심 문제**: 최적화 루프 내에서 JaxSSO `Model` Python 객체를 매 스텝 재빌드하면
Python 오버헤드가 발생하고 jit 추적이 깨진다.

**채택 전략 (Strategy B)**: `K_func`를 pure function으로 직접 호출

```python
from JaxSSO.assemblemodel import K_func

# 설계변수에서 JaxSSO 인자로 변환 (Python side, jit 외부에서 1회)
node_crds  = design_vars_to_node_crds(dv, base_model)   # (N, 3)
prop_quads = design_vars_to_prop_quads(dv, base_model)  # (M, 3) [t, E, nu]

# 이 함수만 jit 안에 들어감
@jax.jit
def _loss_fn(node_crds, prop_quads):
    K = K_func(node_crds, ndof, n_bc, cnct_bc, prop_bc,
               n_quad, cnct_quads, prop_quads)
    ...
    return loss
```

**장점**:
- JaxSSO Python 객체 재빌드 없음
- `jax.grad(_loss_fn)` 직접 적용 가능
- vmap으로 복수 하중 케이스 병렬화 가능

---

### 4.3 모달 자동미분 전략 [NEW v1.1]

**문제**: `scipy.sparse.linalg.eigsh`는 JAX autodiff 체인을 끊는다.

**채택 전략**: `jnp.linalg.eigh` (dense) + 고유값 감도식

```python
# Dense eigenvalue decomposition (JAX autodiff 호환)
eigenvalues, eigenvectors = jnp.linalg.eigh(K_dense, M_dense)
# K @ phi = lambda * M @ phi
# eigh는 일반화 고유치 문제를 Cholesky factoring M 후 표준화하여 풀음

# 자동미분이 필요 없는 경우 (타겟 주파수만 비교):
# scipy.sparse.linalg.eigsh 사용 가능 (더 빠름)
# JAX 손실함수에서 타겟으로만 사용하고 grad는 취하지 않음

# 자동미분이 필요한 경우 (주파수 최적화):
# custom_vjp wrapper 사용
@jax.custom_vjp
def modal_solve(K, M):
    vals, vecs = jnp.linalg.eigh(K, M)
    return vals, vecs

def modal_solve_fwd(K, M):
    vals, vecs = modal_solve(K, M)
    return (vals, vecs), (K, M, vals, vecs)

def modal_solve_bwd(res, g):
    K, M, vals, vecs = res
    g_vals, g_vecs = g
    # 고유값 감도식: dλ/dp = φᵀ(dK/dp - λ dM/dp)φ
    # g_vals는 각 λ에 대한 업스트림 그래디언트
    dK = sum(g_vals[i] * jnp.outer(vecs[:,i], vecs[:,i])
             for i in range(len(vals)))
    dM = sum(-g_vals[i] * vals[i] * jnp.outer(vecs[:,i], vecs[:,i])
             for i in range(len(vals)))
    return dK, dM

modal_solve.defvjp(modal_solve_fwd, modal_solve_bwd)
```

**실용 판단**:
- 모달 해석 자체는 dense이므로 모델 크기에 주의 (N < 5000 권장)
- 더 큰 모델은 sparse eigensolver로 타겟 계산, dense로 grad 계산 분리

---

### 4.4 반력 계산 전략 [NEW v1.1 — 전략 변경]

**v1.0 전략 (변경 전)**: `R = K·u - F_ext` (후처리 계산)

**v1.1 전략 (채택)**: JaxSSO augmented solver의 Lagrange multiplier 직접 사용

```python
# JaxSSO solver.py 내부 구조 (참조용)
# u_aug = [u_free (ndof,), λ (n_bc,)]
# u_aug[ndof:] = Lagrange multipliers = BC 노드에서의 반력

u_aug = jax_sparse_solve(K_aug, f_aug)  # augmented system
u_free = u_aug[:ndof]    # 자유도 변위
lambda_ = u_aug[ndof:]   # Lagrange multiplier = 반력
```

**반력 API**:

```python
class WHTSolverResult:
    _u_aug: np.ndarray       # (ndof + n_bc,) augmented solution 전체 보관
    _bc_node_ids: list[int]  # BC 노드 ID (순서 보장)
    _bc_dofs: list[int]      # 각 BC 노드의 구속 DOF

    def reaction_force(
        self,
        node_ids: int | list[int] | None = None,
    ) -> np.ndarray:
        """
        u_aug[ndof:] 에서 Lagrange multiplier 값 반환.

        Parameters
        ----------
        node_ids : None → BC 노드 전체 (n_bc_nodes, 3) 반환
                   int  → 해당 노드 (3,) [Rx, Ry, Rz]
                   list → 선택 노드 (len, 3)

        Returns: 병진 반력 [Rx, Ry, Rz]만 반환 (회전 반력 제외)

        사용 예 (최적화 제약):
            r = result.reaction_force(support_set_nodes)
            total_Rz = jnp.sum(r[:, 2])   # Z 방향 합력 ≈ 총 하중
        """
        lambda_ = self._u_aug[self._ndof:]
        # BC DOF에서 노드별 반력 재조립
        ...
```

**v1.0 대비 장점**:
- K와 u를 별도로 계산할 필요 없음
- Lagrange multiplier는 augmented solve와 동시에 얻어짐
- JAX autodiff 체인 유지 (제약함수로 직접 사용 가능)

---

### 4.5 모드 스위칭 해결 전략 [NEW v1.1]

**문제**: 최적화 과정에서 모드 순서가 바뀌면 `freq[0]_opt ↔ freq[1]_target` 잘못 매칭.

**전략**: MAC 행렬 기반 soft assignment

```python
def mac_matrix(
    phis_opt:    jnp.ndarray,    # (n_modes_opt, N) 최적화 모델 모드벡터
    phis_target: jnp.ndarray,    # (n_modes_tgt, N) 타겟 모드벡터 (매핑 후)
) -> jnp.ndarray:                # (n_modes_opt, n_modes_tgt) MAC 행렬
    """
    MAC[i,j] = (φ_opt[i] · φ_tgt[j])² / ((φ_opt[i]·φ_opt[i])(φ_tgt[j]·φ_tgt[j]))
    """

def freq_loss_with_mac_assignment(
    freqs_opt:    jnp.ndarray,   # (n_modes,)
    freqs_target: jnp.ndarray,   # (n_modes,)
    phis_opt:     jnp.ndarray,   # (n_modes, N)
    phis_target:  jnp.ndarray,   # (n_modes, N)
) -> jnp.ndarray:
    """
    MAC 행렬로 모드 대응 관계를 구한 후 주파수 오차 계산.

    1. MAC_mat = mac_matrix(phis_opt, phis_target)  → (n×m)
    2. assignment = argmax(MAC_mat, axis=1)          → opt[i] ↔ target[assignment[i]]
    3. loss = Σ (freqs_opt[i] - freqs_target[assignment[i]])²

    Soft 버전 (autodiff 가능):
    - argmax 대신 MAC_mat을 가중치로 soft-assignment
    - loss = Σ_i Σ_j MAC_mat[i,j] * (freqs_opt[i] - freqs_target[j])²
    """
```

---

### 4.6 토포그래피 부드러움 정규화 [NEW v1.1]

**문제**: z_offsets가 인접 노드 간 급격히 변화하면 해석 불안정 + 제조 불가.

**전략**: Laplacian 정규화 항 추가

```python
def laplacian_smoothness(
    z_offsets: jnp.ndarray,     # (N,) 노드별 Z-offset
    adjacency: jnp.ndarray,     # (N, max_neighbors) 인접 노드 인덱스 (패딩 -1)
    lambda_smooth: float = 0.01,
) -> jnp.ndarray:
    """
    ||L @ z||² 형태의 Laplacian 정규화.

    L = 그래프 Laplacian (D - A)
    D[i,i] = 노드 i의 이웃 수
    A[i,j] = 1 if (i,j) 연결, else 0

    loss_total = loss_structural + lambda_smooth * laplacian_smoothness(z, adj)

    lambda_smooth 튜닝:
    - 0.001: 약한 스무딩 (거친 토포그래피 허용)
    - 0.01:  중간 (권장 시작값)
    - 0.1:   강한 스무딩 (완만한 형상만 허용)
    """
    # 구현: (z[i] - mean(z[neighbors(i)]))² 합산
    diffs = z_offsets[adjacency] - z_offsets[:, None]  # 패딩 처리 필요
    return lambda_smooth * jnp.sum(diffs ** 2)
```

---

### 4.7 `WHTSolver` — JaxSSO 래퍼

```python
class WHTSolver:
    """
    WHTMeshModel을 받아 JaxSSO 모델을 빌드하고 해석을 실행한다.
    RBE2를 stiff beam으로 자동 변환한다.

    내부적으로 K_func (pure JAX function)을 추출하여
    최적화 루프에서 Python 객체 재빌드 없이 재사용 가능.
    """

    def __init__(self, model: WHTMeshModel, stiffness_scale: float = 1e3):
        self.model = model
        self.stiffness_scale = stiffness_scale
        # K_func 인자를 한 번 추출하여 캐싱
        self._static_args = self._extract_static_args()

    def solve_static(self, load_case: WHTLoadCase) -> WHTSolverResult:
        """정적 해석. BCs + 하중을 JaxSSO에 적용 후 해석.
        WHTSolverResult._u_aug에 augmented solution 보관 (반력 접근용)."""

    def solve_modal(self, num_modes: int = 10) -> WHTSolverResult:
        """모달 해석. 고유진동수 + 고유벡터 반환.
        jnp.linalg.eigh 사용 (autodiff 가능)."""

    def solve_all(
        self,
        load_cases: list[WHTLoadCase],
        num_modes: int = 10,
    ) -> dict[str, WHTSolverResult]:
        """전체 하중 케이스 + 모달 일괄 실행."""

    def get_k_func_args(self) -> dict:
        """K_func 호출에 필요한 static args 반환.
        WHTOptimizer가 jit 경계 내에서 사용."""
```

---

### 4.8 `WHTSolverResult` — 결과 및 반력 API

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

    # 반력 (Lagrange multiplier 방식)
    _u_aug: np.ndarray          # (ndof + n_bc,) augmented solution 전체
    _ndof: int                  # 자유 DOF 수
    _bc_node_ids: list[int]     # BC 노드 ID 순서
    _bc_dofs: list[int]         # 각 BC 엔트리의 DOF

    def reaction_force(
        self,
        node_ids: int | list[int] | None = None,
    ) -> np.ndarray:
        """
        u_aug[ndof:] (Lagrange multiplier)에서 반력 반환.
        Returns: (N, 3) [Rx, Ry, Rz] 병진 반력
        """

    def to_wht_result_data(
        self,
        metadata: WHTMetadata,
        mesh_model: WHTMeshModel | None = None,
    ) -> WHTResultData:
        """
        wht_converter 포맷으로 변환 → ParaView 출력용.

        mesh_model이 None이 아니면 WHTMeshModel에서 geometry를 우선 사용.
        (설계변수 변경 후 업데이트된 geometry 반영용)
        """
```

---

### 4.9 `WHTMapper` — RBF 보간 (Hi-fi → Lo-fi 매핑)

```python
class WHTMapper:
    """
    서로 다른 메시 밀도의 두 모델 간 결과 데이터를 보간한다.
    타겟(Hi-fi) 결과를 최적화 모델(Lo-fi) 노드 위치로 매핑.

    구현: scipy.interpolate.RBFInterpolator (thin-plate spline 기본)
    타겟 변경 지원: re-fit 구조 제공
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

### 4.10 `LoadCaseLibrary` — 하중 케이스 프리셋

```python
class LoadCaseLibrary:
    @staticmethod
    def three_point_bending(model, support_sets, load_target, load_z=-1000.0, constrain_dofs=(0,1,2)) -> WHTLoadCase: ...
    def four_point_bending(model, support_sets, load_targets, load_z=-1000.0) -> WHTLoadCase: ...
    def twisting(model, fixed_corner, twist_corner, load_z=-1000.0) -> WHTLoadCase: ...
    def corner_lift(model, support_sets, lift_target, load_z=1000.0) -> WHTLoadCase: ...
    def end_bending(model, fixed_end, load_end, load_z=-1000.0) -> WHTLoadCase: ...
```

---

### 4.11 `objectives.py` — 목적함수 (JAX 기반)

```python
# ── MAC ──────────────────────────────────────────────────────────────────
def mac(phi_a: jnp.ndarray, phi_b: jnp.ndarray) -> jnp.ndarray:
    """MAC(a,b) = (φ_a·φ_b)² / ((φ_a·φ_a)(φ_b·φ_b))"""

def mass_weighted_mac(phi_a, phi_b, M_diag) -> jnp.ndarray:
    """MAC_M(a,b) = (φ_aᵀMφ_b)² / ((φ_aᵀMφ_a)(φ_bᵀMφ_b))"""

def mac_matrix(phis_opt, phis_target) -> jnp.ndarray:
    """(n_modes_opt, n_modes_tgt) MAC 행렬 — 모드 대응 관계 파악용"""

def freq_loss_with_mac_assignment(freqs_opt, freqs_target, phis_opt, phis_target) -> jnp.ndarray:
    """MAC soft-assignment로 모드 스위칭 문제 해결 후 주파수 MSE 계산"""

def laplacian_smoothness(z_offsets, adjacency, lambda_smooth=0.01) -> jnp.ndarray:
    """Laplacian 정규화: ||L·z||²"""

# ── Multi-objective loss ──────────────────────────────────────────────────
def multi_objective_loss(
    opt_result:    WHTSolverResult,
    target_result: WHTSolverResult,
    mapper:        WHTMapper,
    adjacency:     jnp.ndarray,    # 토포그래피 스무딩용 인접 정보 [NEW v1.1]
    weights: dict = {"freq": 1.0, "mac": 1.0, "static": 1.0, "smooth": 0.01},
    use_mass_mac:  bool = False,
) -> jnp.ndarray:
    """
    f_obj = w1 * freq_loss (MAC soft-assignment 기반)
          + w2 * mac_loss   (1 - MAC 합산)
          + w3 * rmse_loss  (변위 RMSE)
          + w4 * smooth_loss (Laplacian 정규화)
    """
```

---

### 4.12 `WHTOptimizer` — 최적화 엔진

```python
@dataclass
class DesignVariables:
    """JAX pytree로 등록. jax.grad / optax 직접 사용."""
    t_field:   jnp.ndarray   # (M,) 요소별 두께 [mm]
    z_offsets: jnp.ndarray   # (N,) 노드별 Z-offset [mm]
    E:         float          # 전체 탄성계수 [MPa]
    rho:       float          # 전체 밀도 [t/mm³]

@dataclass
class DesignBounds:
    t_min: float;   t_max: float
    z_min: float;   z_max: float
    E_min: float;   E_max: float
    rho_min: float; rho_max: float

class WHTOptimizer:
    def __init__(
        self,
        base_model:     WHTMeshModel,
        target_results: dict[str, WHTSolverResult],
        mapper:         WHTMapper,
        bounds:         DesignBounds,
        load_cases:     list[WHTLoadCase],
        num_modes:      int = 10,
        optimizer:      optax.GradientTransformation = optax.adam(1e-3),
        weights:        dict = {"freq": 1.0, "mac": 1.0, "static": 1.0, "smooth": 0.01},
        monitor:        "OptimizationMonitor | None" = None,   # [NEW v1.1]
    ): ...

    def run(
        self,
        init_vars:  DesignVariables,
        n_steps:    int = 500,
        log_every:  int = 10,
    ) -> tuple[DesignVariables, list[float]]:
        """
        최적화 루프.
        - 매 스텝: grad 계산 → optax 업데이트 → jnp.clip 경계 투영
        - log_every 스텝마다: loss 출력 + monitor.update() 호출
        Returns: (최적 설계변수, loss 히스토리)
        """
```

---

### 4.13 `OptimizationMonitor` (PyVistaQt) [NEW v1.1]

**설계 원칙**: PyVistaQt BackgroundPlotter를 사용하여 Qt 이벤트 루프를
메인 최적화 루프와 독립적인 백그라운드 스레드에서 실행한다.
메인 루프는 `monitor.update()` 직접 호출로 데이터를 밀어 넣는다.

```python
# wht_solver/wht_monitor.py

from pyvistaqt import BackgroundPlotter
import pyvista as pv
import numpy as np
from threading import Lock

class OptimizationMonitor:
    """
    최적화 진행 상태를 PyVistaQt 창에 실시간으로 시각화.

    아키텍처:
    - BackgroundPlotter: Qt 이벤트 루프를 별도 스레드에서 실행
    - 메인 최적화 루프는 update()를 직접 호출
    - 스레드 안전성: pyvistaqt가 내부적으로 Qt Signal/Slot으로 처리

    사용 예:
        monitor = OptimizationMonitor()
        monitor.init_mesh(base_model)
        optimizer = WHTOptimizer(..., monitor=monitor)
        optimizer.run(init_vars, n_steps=500)
    """

    def __init__(
        self,
        update_every: int = 10,        # 시각화 업데이트 주기 (스텝)
        show_loss_plot: bool = True,   # 손실 곡선 서브플롯 표시
    ):
        self._plotter: BackgroundPlotter | None = None
        self._mesh: pv.PolyData | None = None
        self._lock = Lock()
        self._loss_history: list[float] = []
        self._step_history: list[int] = []
        self.update_every = update_every

    def init_mesh(self, model: "WHTMeshModel") -> None:
        """
        최적화 시작 전 초기 메시로 뷰어 창 생성.
        BackgroundPlotter는 이 시점에 Qt 창을 띄움.
        """
        nodes = model.nodes_array()      # (N, 3)
        faces = model.faces_array()      # pv 형식
        self._mesh = pv.PolyData(nodes, faces)

        self._plotter = BackgroundPlotter(title="WHT Optimization Monitor")
        self._plotter.add_mesh(
            self._mesh,
            scalars="z_offset",
            cmap="coolwarm",
            show_edges=True,
            clim=[-10, 10],
        )
        self._plotter.add_text("Step: 0  Loss: —", name="status")
        self._plotter.show()

    def update(
        self,
        step: int,
        nodes: np.ndarray,             # (N, 3) 현재 노드 좌표
        z_offsets: np.ndarray,         # (N,) z-offset scalar field
        loss: float,
    ) -> None:
        """
        메인 루프에서 직접 호출.
        BackgroundPlotter.update()로 Qt 창 갱신.
        스레드 안전 (pyvistaqt 내부 Qt Signal 사용).
        """
        if self._plotter is None or self._mesh is None:
            return

        self._mesh.points = nodes
        self._mesh["z_offset"] = z_offsets
        self._loss_history.append(loss)
        self._step_history.append(step)

        self._plotter.update_scalars(z_offsets, mesh=self._mesh)
        self._plotter.update_text(
            f"Step: {step}  Loss: {loss:.4f}",
            name="status",
        )
        self._plotter.update()

    def close(self) -> None:
        """최적화 완료 후 창 유지 (사용자가 수동으로 닫을 수 있음)."""
        if self._plotter is not None:
            # 창을 닫지 않고 유지 — 최종 상태 확인용
            pass
```

**메인 루프 통합**:

```python
# WHTOptimizer.run() 내부 (개략)
for step in range(n_steps):
    loss, grads = jax.value_and_grad(loss_fn)(current_vars)
    updates, opt_state = optimizer.update(grads, opt_state)
    current_vars = optax.apply_updates(current_vars, updates)
    current_vars = clip_to_bounds(current_vars, bounds)

    if step % log_every == 0:
        print(f"Step {step}: loss={float(loss):.4f}")
        if self.monitor is not None:
            self.monitor.update(
                step=step,
                nodes=get_current_nodes(current_vars),
                z_offsets=np.array(current_vars.z_offsets),
                loss=float(loss),
            )
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

# ── Step 2. 최적화 모델 생성 (단순 트레이) ────────────────────────────
opt_model = WHTMeshModel.from_node_elem_db(*generate_shell_tray(...))

# ── Step 3. 노드 매핑 준비 ───────────────────────────────────────────────
mapper = WHTMapper()
mapper.fit(
    source_nodes = target_model.nodes_array(),
    source_data  = target_results["modal"].mode_shapes[0],
)

# ── Step 4. 모니터 초기화 ────────────────────────────────────────────────
monitor = OptimizationMonitor(update_every=10)
monitor.init_mesh(opt_model)   # Qt 창 열림

# ── Step 5. 최적화 실행 ─────────────────────────────────────────────────
bounds = DesignBounds(t_min=0.5, t_max=5.0, z_min=0.0, z_max=10.0,
                      E_min=500, E_max=5000, rho_min=5e-10, rho_max=5e-9)
init_vars = DesignVariables(
    t_field   = jnp.full(opt_model.n_elements, 2.0),
    z_offsets = jnp.zeros(opt_model.n_nodes),
    E=1000.0, rho=1e-9,
)
optimizer = WHTOptimizer(
    base_model=opt_model, target_results=target_results,
    mapper=mapper, bounds=bounds, load_cases=[...],
    weights={"freq": 2.0, "mac": 1.0, "static": 1.0, "smooth": 0.01},
    monitor=monitor,
)
best_vars, loss_history = optimizer.run(init_vars, n_steps=500)

# ── Step 6. 결과 출력 ───────────────────────────────────────────────────
final_model = opt_model.apply_design(best_vars)
final_solver = WHTSolver(final_model)
final_result = final_solver.solve_modal()
rd = final_result.to_wht_result_data(meta, mesh_model=final_model)
VTKHDFExporter().export(rd, "results/optimized.hdf")
```

---

## 6. 구현 로드맵 + Phase별 스모크 테스트 [v1.1 업데이트]

### Phase 1 — `wht_modeler` Core

- [ ] `wht_entities.py`: WHTNode, WHTElement, WHTSet, WHTRBE2, WHTProperty, WHTMaterial
- [ ] `wht_mesh_model.py`: WHTMeshModel (CRUD API + apply_spc/apply_force)
- [ ] `tests/test_mesh_model.py`

**스모크 테스트**:
```python
# exam1_nf.py의 node_db/elem_db를 WHTMeshModel로 변환
model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
assert len(model.nodes) == len(node_db)
assert len(model.elements) == len(elem_db)
model.apply_spc([0,1,2], dofs=(0,1,2,3,4,5))
rd = model.to_wht_result_data()
assert rd.n_nodes == len(node_db)
```

---

### Phase 2 — LS-DYNA IO

- [ ] `io/lsdyna_reader.py`: *NODE, *ELEMENT_SHELL, *SET_NODE_LIST/GENERAL, RBE2
- [ ] `io/lsdyna_writer.py`: WHTMeshModel → .k 파일
- [ ] `tests/test_lsdyna_io.py`: 왕복 변환 테스트

**스모크 테스트**:
```python
# 왕복 변환: read → write → re-read → 동일성 확인
model_orig = LSDYNAReader().read("test_tray.k")
LSDYNAWriter().write(model_orig, "/tmp/tray_out.k")
model_back = LSDYNAReader().read("/tmp/tray_out.k")
assert set(model_orig.nodes.keys()) == set(model_back.nodes.keys())
# 노드 좌표 허용 오차 내 일치
np.testing.assert_allclose(
    model_orig.nodes_array(), model_back.nodes_array(), rtol=1e-6
)
```

---

### Phase 3 — `wht_solver` Core

- [ ] `wht_solver.py`: WHTSolver (K_func 직접 호출 Strategy B)
- [ ] `wht_result.py`: WHTSolverResult + reaction_force() (Lagrange multiplier)
- [ ] `load_cases.py`: LoadCaseLibrary 5종

**스모크 테스트**:
```python
# exam1_nf.py와 동일한 모달 결과가 나오는지 확인
solver = WHTSolver(model)
result = solver.solve_modal(num_modes=6)
assert result.frequencies.shape == (6,)
assert result.mode_shapes.shape == (6, n_nodes, 6)

# 정적 해석 + 반력 확인
lc = LoadCaseLibrary.three_point_bending(model, support_sets=[...], load_target=[...])
static_result = solver.solve_static(lc)
r = static_result.reaction_force()          # (n_bc_nodes, 3)
total_Rz = np.sum(r[:, 2])
np.testing.assert_allclose(total_Rz, -load_z, rtol=1e-4)  # 힘 평형
```

---

### Phase 4 — Mapper + MAC

- [ ] `wht_mapper.py`: WHTMapper (RBF, re-fit 지원)
- [ ] `objectives.py`: mac(), mass_weighted_mac(), mac_matrix(), freq_loss_with_mac_assignment(), laplacian_smoothness(), multi_objective_loss()
- [ ] `tests/test_mapper.py`

**스모크 테스트**:
```python
# 같은 메시에서 매핑 → 거의 동일해야 함
mapper = WHTMapper()
mapper.fit(model.nodes_array(), result.mode_shapes[0])
mapped = mapper.transform(model.nodes_array())
np.testing.assert_allclose(mapped, result.mode_shapes[0], rtol=1e-3)

# MAC: 동일 벡터 → 1.0
phi = result.mode_shapes[0].reshape(-1)
assert abs(float(mac(phi, phi)) - 1.0) < 1e-6

# MAC 행렬: 모드 대각선이 최대
mac_mat = mac_matrix(result.mode_shapes.reshape(n_modes, -1),
                     result.mode_shapes.reshape(n_modes, -1))
assert jnp.all(jnp.argmax(mac_mat, axis=1) == jnp.arange(n_modes))
```

---

### Phase 5 — Optimizer + Monitor + 나머지 IO

- [ ] `wht_optimizer.py`: DesignVariables, DesignBounds, WHTOptimizer (Optax Adam)
- [ ] `wht_monitor.py`: OptimizationMonitor (PyVistaQt BackgroundPlotter)
- [ ] `io/optistruct_reader.py`, `io/radioss_reader.py`, `io/abaqus_reader.py`
- [ ] `tests/test_optimizer.py`

**스모크 테스트**:
```python
# 10스텝만 실행해서 loss가 감소하는지 확인
best_vars, history = optimizer.run(init_vars, n_steps=10)
assert history[-1] < history[0], "Loss must decrease"

# 경계 조건 위반 없음
assert jnp.all(best_vars.t_field >= bounds.t_min)
assert jnp.all(best_vars.t_field <= bounds.t_max)
assert jnp.all(best_vars.z_offsets >= bounds.z_min)
```

---

## 7. 핵심 기술 결정 (v1.1 업데이트)

| 항목 | 결정 | 근거 |
|------|------|------|
| JAX jit 경계 | K_func 직접 호출 (Strategy B) | assemblemodel.K_func는 pure JAX function (JaxSSO 소스 확인) |
| RBE2 구현 | Stiff Beam (stiffness_scale 옵션) | JaxSSO 수정 불필요, JAX autodiff 안정 |
| RBE2 기본 강성비 | `1e3` (10^3) | cond(K) < 10^12 유지 |
| 반력 계산 | u_aug[ndof:] (Lagrange multiplier) | JaxSSO solver.py 확인: u_aug = [u_free, λ] |
| 반력 API | reaction_force(node_ids) → (N,3) | 최적화 제약함수와 직접 연동 |
| 모달 autodiff | jnp.linalg.eigh + 고유값 감도식 custom_vjp | scipy eigsh는 autodiff 불가 |
| 모드 스위칭 | MAC 행렬 soft-assignment | argmax 대신 MAC 가중치로 미분 가능한 대응 |
| 토포그래피 스무딩 | Laplacian 정규화 (lambda_smooth=0.01) | 인접 노드 간 z-offset 변화 억제 |
| MAC 기본 | 표준 벡터 내적 (RBF 매핑 후) | 단순, JAX jit 호환 |
| MAC 옵션 | mass_weighted_mac() 별도 제공 | 에너지 기반 비교가 필요할 때 |
| 최적화 | Optax Adam 기본 | JaxSSO 자동미분 생태계, bounds는 jnp.clip 투영 |
| 보간법 | scipy RBFInterpolator (thin-plate spline) | 전처리 1회, 루프 중 재사용 |
| 최적화 모니터링 | PyVistaQt BackgroundPlotter | Qt 이벤트 루프 독립 스레드, 콜백으로 업데이트 |
| WHTResultData 변환 | to_wht_result_data(metadata, mesh_model=) | geometry 업데이트 반영 가능 |
| SET 지원 | LIST + GENERAL(BOX, PART) | 사용자 요구사항 |

---

## 8. JaxSSO 사용 전략 요약

```
[최적화 루프 내 — jit 적용 대상]
K_func(node_crds, ndof, n_bc, cnct_bc, prop_bc, n_quad, cnct_quads, prop_quads)
    ↑
DesignVariables(t_field, z_offsets, E, rho)
    → node_crds = base_crds + [0, 0, z_offsets]
    → prop_quads[:, 0] = t_field   (두께)
    → prop_quads[:, 1] = E         (탄성계수)

[최적화 루프 외 — 1회 실행]
WHTMeshModel → JaxSSO Model 빌드
→ ndof, n_bc, cnct_bc, prop_bc, n_quad, cnct_quads 추출 (static args)
→ 이 값들은 루프 중 변하지 않음 (topology 고정)

[JaxSSO SSO_model 사용 안 함]
→ SSO_model은 Python 객체 의존성 있음
→ K_func만 추출하여 직접 사용
```

---

*이 계획은 구현 시작 전 검토·승인 대상입니다.*
