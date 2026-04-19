# [Design Proposal] WHT Universal FEM Result Converter

**Date**: 2026-04-18
**Version**: 0.5 (Implementation-Synchronized)
**Status**: Phase 1–3 Complete / Phase 4 Partial
**Target**: Commercial-grade FEM Post-processing Engine
**Target Solvers**: JaxSSO (Static / Modal / Buckling), jax-fem (Static / Transient / Modal)
**Target Viewers**: ParaView 5.11+, Altair HyperView

> **v0.4 → v0.5 변경 요약**
> - 구현 코드(wht_converter/)와 설계 내용 동기화 (코드가 기준)
> - wht_utils.py 유틸리티 함수 전체 문서화 (merge_csr, node_dict_to_array, remap_connectivity 추가)
> - VTKHDF 스키마에 Steps/PartOffsets 추가 (ParaView 6 호환 필수 필드)
> - WHTResultData 검증: assert → WHTValidationError 예외로 수정
> - WHTMetadata __post_init__ 검증 로직 문서화
> - HWASCIIExporter: 단일 SUPPORTED_BLOCKS → SUPPORTED_POINT_BLOCKS / SUPPORTED_CELL_BLOCKS 분리
> - CLI: positional script → --input/-i 플래그로 수정, --dry-run 등 추가 옵션 문서화
> - VTUPVDExporter: meshio 미사용, 직접 XML 작성으로 확정
> - 구현 현황(Phase 체크리스트) 반영
> - __main__.py 절대 import → 상대 import 버그 수정

---

## 0. 파일 구조 (구현 기준)

```
wht_converter/
├── __init__.py          ← 공개 API 전체 re-export
├── __main__.py          ← CLI 엔트리포인트 (python -m wht_converter)
├── wht_models.py        ← WHTMetadata, WHTResultData, 예외 클래스
├── wht_utils.py         ← VTKCellType, to_vtk_csr, merge_csr,
│                            node_dict_to_array, remap_connectivity
├── wht_adapters.py      ← BaseAdapter, JaxSSOAdapter, JaxFEMAdapter
├── wht_exporters.py     ← BaseExporter, VTKHDFExporter, VTUPVDExporter,
│                            HWASCIIExporter
└── tests/
    ├── test_models_and_utils.py
    ├── test_adapters.py
    └── test_exporters.py

test_jaxSSO/
└── exam1_nf.py          ← ParaViewExporter 클래스로 wht_converter 호출
```

---

## 1. Overview & Design Principles

JAX 기반 FEM 솔버(JaxSSO, jax-fem)와 산업용 후처리 도구(ParaView, HyperView) 사이의
명확하고 결합도 낮은 변환 인터페이스를 제공한다.

### 1.1 Design Principles

| 원칙 | 설명 |
|------|------|
| **Single Responsibility** | 데이터 컨테이너·어댑터·익스포터는 각각 독립 교체 가능 |
| **Open/Closed** | 새 솔버·포맷 추가 시 기존 코드 수정 없이 서브클래스만 작성 |
| **Explicit Validation** | 각 단계 경계에서 shape·단위·타입을 명시적으로 검사 |
| **Fail Fast** | 오류는 가능한 한 조기에 발생시켜 후처리 단계의 조용한 버그를 방지 |

### 1.2 Data Flow

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Solver Native  │     │  Adapter Layer        │     │  WHTResultData (IR)  │
│  ─────────────  │     │  ──────────────────── │     │  ──────────────────  │
│  JaxSSO:        │────▶│  JaxSSOAdapter        │────▶│  nodes    (N, 3)     │
│   model.nodes   │     │  .convert(model,      │     │  connectivity (K,)   │
│   model.quads   │     │           results,    │     │  offsets  (M+1,)     │
│   freqs, vecs   │     │           "modal",    │     │  cell_types (M,)     │
│                 │     │           meta)       │     │  point_data {T,N,D}  │
│  jax-fem:       │     │                       │     │  cell_data  {T,M,D}  │
│   problem.mesh  │────▶│  JaxFEMAdapter        │     │  time_values (T,)    │
│   u, t, eigvecs │     │  .convert(problem,    │     │  metadata            │
│                 │     │           results,    │     └──────────┬───────────┘
└─────────────────┘     │           "transient",│                │
                        │           meta)       │                ▼
                        └──────────────────────┘     ┌──────────────────────┐
                                                      │  Exporter Layer      │
                                                      │  ──────────────────  │
                                                      │  VTKHDFExporter      │
                                                      │    → .hdf (단일파일) │
                                                      │  VTUPVDExporter      │
                                                      │    → .pvd + N×.vtu  │
                                                      │  HWASCIIExporter     │
                                                      │    → .ascii          │
                                                      └──────────────────────┘
```

---

## 2. Core Data Models (`wht_models.py`)

### 2.1 `WHTMetadata` — 솔버/해석 메타정보

```python
@dataclass
class WHTMetadata:
    solver_name: str          # "JaxSSO" | "jax-fem"
    solver_version: str       # 예: "0.1.0"
    analysis_type: str        # "static" | "modal" | "transient" | "buckling"
    coordinate_system: str    # "cartesian" | "cylindrical"
    unit_length: str          # "m" | "mm" | "in"
    unit_force: str           # "N" | "kN" | "lbf"
    created_at: str = ""      # ISO 8601; 빈 문자열이면 __post_init__에서 자동 생성
```

**`__post_init__` 검증 (구현됨)**  
아래 필드는 허용 집합 외의 값을 넘기면 즉시 `WHTValidationError`를 발생시킨다.

| 필드 | 허용 값 |
|------|---------|
| `analysis_type` | `{"static", "modal", "transient", "buckling"}` |
| `coordinate_system` | `{"cartesian", "cylindrical"}` |
| `unit_length` | `{"m", "mm", "in"}` |
| `unit_force` | `{"N", "kN", "lbf"}` |

> ⚠️ **단위 불일치 방지**: mm vs m 혼용 시 HyperView에서 스케일 1000배 오차가 발생한다.
> 기본값이 없으므로 어댑터는 반드시 `unit_length`를 명시적으로 전달해야 한다.

---

### 2.2 `WHTResultData` — 중간 표현(IR)

모든 솔버 출력과 익스포터 입력의 **단일 교환 포맷**.  
VTK CSR(Compressed Sparse Row) flat 포맷을 채택하여 혼합 요소(빔+셸+솔리드) 메시를 정확히 표현한다.

```python
@dataclass
class WHTResultData:
    # --- Geometry (CSR Flat Format) ---
    nodes:        np.ndarray   # (N, 3) float64  — 노드 좌표 [x, y, z]
    connectivity: np.ndarray   # (K,)   int64    — 플랫 노드 인덱스 배열
    offsets:      np.ndarray   # (M+1,) int64    — offsets[i]:offsets[i+1] = i번째 셀의 노드 행
    cell_types:   np.ndarray   # (M,)   uint8    — VTK 셀 타입 정수 상수

    # --- Named Sets ---
    node_sets:    dict[str, np.ndarray]   # {"fixed": [0,1,2]} — 경계조건 그룹 등
    element_sets: dict[str, np.ndarray]   # {"beam_group": [5,6,7]}

    # --- Results (모두 T 축이 선두) ---
    point_data: dict[str, np.ndarray]  # {name: (T, N, D)}  — 노달 결과
    cell_data:  dict[str, np.ndarray]  # {name: (T, M, D)}  — 요소 결과
    field_data: dict[str, np.ndarray]  # {name: (T,)}       — 전역 스칼라 (예: 하중계수)

    # --- T 축 의미 (analysis_type별) ---
    time_values: np.ndarray  # (T,)
    #   static:    [0.0]                     T=1
    #   transient: [t0, t1, ..., tT-1]  [s] T=실제 타임스텝 수
    #   modal:     [f0, f1, ..., fn-1]   [Hz] T=모드 수
    #   buckling:  [λ0, λ1, ..., λn-1]       T=모드 수

    metadata: WHTMetadata
```

**`__post_init__` 검증 (구현됨)**  
생성 즉시 아래를 검사하며, 실패하면 `WHTValidationError`를 즉시 발생시킨다.

| 조건 | 오류 메시지 패턴 |
|------|----------------|
| `nodes.shape == (N, 3)` | `"nodes must be (N, 3)"` |
| `len(offsets) == M + 1` | `"offsets must be 1-D array of length M+1"` |
| `len(cell_types) == M` | `"cell_types length must equal M"` |
| `len(connectivity) == offsets[-1]` | `"connectivity length ... does not match"` |
| `point_data[name].shape == (T, N, D)` | `"point_data[...].shape[0/1] != T/N"` |
| `cell_data[name].shape == (T, M, D)` | `"cell_data[...].shape[0/1] != T/M"` |
| `field_data[name].shape == (T,)` | `"field_data[...] must be shape (T,)"` |

**편의 프로퍼티**

```python
data.n_nodes      # → N
data.n_cells      # → M
data.n_timesteps  # → T
```

---

### 2.3 CSR Flat Format 상세

혼합 요소 메시에서 모든 셀을 하나의 배열에 연속 저장하는 방식.  
VTK, ParaView, h5py VTKHDF 모두 이 포맷을 직접 지원한다.

```
예: Beam(2노드) 1개 + Quad(4노드) 1개 + Hexa(8노드) 1개 혼합 메시

connectivity = [0, 1,   2, 3, 4, 5,   6, 7, 8, 9, 10, 11, 12, 13]
offsets      = [0, 2,   6,             14]          ← 길이 M+1 = 4
cell_types   = [3,      9,             12]           ← 길이 M = 3
               └─LINE   └─QUAD          └─HEXAHEDRON
```

---

### 2.4 예외 클래스

```python
class WHTValidationError(Exception):
    """
    데이터 구조·shape·단위·단조성 오류.
    파이프라인을 즉시 중단한다. 절대 조용히 catch하지 말 것.
    """

class WHTExportWarning(UserWarning):
    """
    익스포터가 미지원 필드를 스킵할 때 발생.
    경고 후 익스포트는 계속 진행된다.
    """
```

---

## 3. Utility Layer (`wht_utils.py`)

### 3.1 `VTKCellType` — VTK 셀 타입 상수

```python
class VTKCellType:
    LINE         =  3   # 2-node beam / truss
    TRIANGLE     =  5   # 3-node triangle
    QUAD         =  9   # 4-node quadrilateral (shell, MITC4)
    TETRA        = 10   # 4-node tetrahedron
    HEXAHEDRON   = 12   # 8-node hexahedron (brick)
    WEDGE        = 13   # 6-node wedge / prism
    PYRAMID      = 14   # 5-node pyramid
```

---

### 3.2 `to_vtk_csr()` — 단일 요소 타입 → CSR

모든 셀이 동일한 노드 수 V를 가질 때 사용.

```python
def to_vtk_csr(
    elements: np.ndarray,   # (M, V) — M개 셀, 셀당 V개 노드
    vtk_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    connectivity : (M*V,)  int64
    offsets      : (M+1,)  int64   ← offsets[0]=0, offsets[i]=i*V
    cell_types   : (M,)    uint8
    """
```

**사용 예**:
```python
quads = np.array([[0,1,2,3],[4,5,6,7]])
conn, offs, types = to_vtk_csr(quads, VTKCellType.QUAD)
# conn  = [0,1,2,3,4,5,6,7]
# offs  = [0, 4, 8]
# types = [9, 9]
```

---

### 3.3 `merge_csr()` — 복수 CSR 그룹 병합

서로 다른 요소 타입(빔+셸 등)을 하나의 CSR 메시로 합칠 때 사용.  
각 그룹은 `to_vtk_csr()` 출력 형태여야 한다 (offsets[0]=0 가정).

```python
def merge_csr(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    groups : [(connectivity, offsets, cell_types), ...]
        각 tuple은 to_vtk_csr() 반환값. offsets는 반드시 0에서 시작.

    Returns
    -------
    connectivity : (K_total,)    — 모든 그룹 연결 배열 이어붙이기
    offsets      : (M_total+1,)  — 전역 오프셋 (그룹 경계에서 누적)
    cell_types   : (M_total,)
    """
```

**사용 예**:
```python
beams = to_vtk_csr(beam_conn, VTKCellType.LINE)   # beam_conn: (B, 2)
quads = to_vtk_csr(quad_conn, VTKCellType.QUAD)   # quad_conn: (Q, 4)
conn, offs, types = merge_csr([beams, quads])
# offs = [0, 2, ..., 2*B, 2*B+4, ..., 2*B+4*Q]
```

---

### 3.4 `node_dict_to_array()` — JaxSSO 노드 dict → ndarray

JaxSSO `model.nodes` 형태의 dict를 정렬된 0-based ndarray로 변환한다.

```python
def node_dict_to_array(
    node_dict: dict,   # {node_id (int): [x, y, z]}
) -> tuple[np.ndarray, dict]:
    """
    Returns
    -------
    nodes  : (N, 3) float64  — node_id 오름차순 정렬
    id_map : dict            — {original_node_id: 0-based_row_index}

    Note
    ----
    node_id가 연속적이지 않아도 동작 (예: {10: ..., 20: ..., 30: ...}).
    id_map을 remap_connectivity()에 전달하여 요소 연결 배열 재인덱싱.
    """
```

---

### 3.5 `remap_connectivity()` — 요소 dict → 0-based 2D 배열

`node_dict_to_array()` 의 `id_map`을 사용하여 요소 연결 배열을 0-based로 재인덱싱한다.

```python
def remap_connectivity(
    elem_dict: dict,   # {elem_id (int): [node_id_0, node_id_1, ...]}
    id_map:    dict,   # node_dict_to_array()의 반환값
) -> np.ndarray:       # (M, V) int64 — elem_id 오름차순 정렬
```

---

## 4. Adapter Layer (`wht_adapters.py`)

### 4.1 `BaseAdapter` — 추상 기반 클래스

```python
class BaseAdapter(ABC):

    @abstractmethod
    def convert(self, *args, **kwargs) -> WHTResultData:
        """솔버 네이티브 데이터를 WHTResultData (IR)로 변환한다."""

    def validate(self, data: WHTResultData) -> None:
        """
        __post_init__ 이후 의미론적 검증.

        분석 유형별 규칙
        ----------------
        transient : time_values가 strictly monotonically increasing인지 확인.
                    → 위반 시 WHTValidationError
        modal     : time_values (주파수) 가 모두 양수인지 확인.
                    → 위반 시 WHTValidationError
        buckling  : time_values (하중계수) 에 0 이하가 있으면 WHTExportWarning 경고.
                    → 변환은 계속 진행됨
        """
```

---

### 4.2 `JaxSSOAdapter`

#### 지원 분석 유형

| `analysis_type` | `results` dict 필수 키 | `time_values` | `point_data` 키 |
|-----------------|----------------------|---------------|-----------------|
| `"static"`   | `"u"` (N, D)                                | `[0.0]`        | `"Displacement"` |
| `"modal"`    | `"vecs"` (n_modes, N, D), `"freqs"` (n_modes,) [Hz] | freqs   | `"Displacement"` |
| `"buckling"` | `"modes"` (n_modes, N, D), `"load_factors"` (n_modes,) | load_factors | `"BucklingMode"` |

> **"transient"는 지원하지 않음** — JaxSSO는 transient 해석을 제공하지 않는다.

#### 확인된 JaxSSO model 속성 (duck-typing)

| 속성 | 타입 | 내용 |
|------|------|------|
| `model.nodes`    | `dict {int: [x,y,z]}`       | 노드 좌표. ID가 비연속적일 수 있음 |
| `model.quads`    | `dict {int: [n0,n1,n2,n3]}` | MITC4 shell 요소 → VTK_QUAD (9) |
| `model.beamcols` | `dict {int: [n0,n1]}`       | 빔/기둥 요소 → VTK_LINE (3) |
| `model.truss`    | `dict {int: [n0,n1]}`       | 트러스 요소 → VTK_LINE (3) |

> `exam1_nf.py`처럼 `model` 객체 대신 별도의 `node_db`/`elem_db`를 쓰는 경우,
> `nodes`·`quads` 속성을 가진 proxy 객체를 만들어 전달한다.
> ```python
> class _ModelProxy:
>     def __init__(self, node_db, elem_db):
>         self.nodes = node_db   # {nid: [x,y,z]}
>         self.quads = elem_db   # {eid: [n0,n1,n2,n3]}
>         # beamcols, truss는 없으면 hasattr 체크로 스킵됨
> ```

#### 내부 호출 흐름

```
JaxSSOAdapter.convert(model, results, analysis_type, metadata)
    │
    ├─ node_dict_to_array(model.nodes)        → nodes (N,3), node_id_map
    ├─ _build_mesh(model, node_id_map)
    │    ├─ remap_connectivity(model.quads, id_map) → to_vtk_csr(..., QUAD)
    │    ├─ remap_connectivity(model.beamcols, ...) → to_vtk_csr(..., LINE)  [있으면]
    │    ├─ remap_connectivity(model.truss, ...)    → to_vtk_csr(..., LINE)  [있으면]
    │    └─ merge_csr([quad_group, beam_group, ...])
    │
    ├─ _convert_static / _convert_modal / _convert_buckling
    │    → point_data, cell_data, field_data, time_values
    │
    ├─ WHTResultData(...)        ← __post_init__ shape 검증
    └─ self.validate(data)       ← 단조성·부호 검증
```

#### 미해결 항목

- [ ] `model.quads` 노드 순서가 VTK CCW(Counter-Clockwise) 와인딩 규약과 일치하는지 확인
- [ ] Buckling `load_factors` 정렬 순서 (오름차순 가정 여부)

---

### 4.3 `JaxFEMAdapter`

#### 지원 분석 유형

| `analysis_type` | `results` dict 필수 키 | `time_values` | `point_data` 키 |
|-----------------|----------------------|---------------|-----------------|
| `"static"`    | `"u"` (N, D)                                    | `[0.0]`     | `"Displacement"` |
| `"transient"` | `"u"` (T, N, D), `"t"` (T,) [s]               | t 배열      | `"Displacement"` |
| `"modal"`     | `"eigvecs"` (n_modes, N, D), `"eigvals"` (n_modes,) [rad²/s²] | freqs (Hz 변환) | `"Displacement"` |

> **"buckling"은 지원하지 않음** — jax-fem은 buckling 해석을 제공하지 않는다.
>
> `eigvals` 단위: rad²/s² 가정. `freqs = sqrt(|eigvals|) / (2π)` 로 변환한다.

#### 확인된 jax-fem problem 속성

| 속성 | 타입 | 내용 |
|------|------|------|
| `problem.mesh.node_coords` | JAX DeviceArray (N, D) | D=2이면 Z=0으로 패딩하여 (N,3) 생성 |
| `problem.mesh.cells` | np.ndarray (M, V) | cells, elements, connect 등 여러 후보 중 자동 탐색 |

#### 미해결 항목

- [ ] `problem.mesh` 요소 연결 배열 필드명 확정 (현재 `cells`, `elements`, `connect`, `cell_connectivity`, `connectivity`, `elem_conn` 순서로 탐색)
- [ ] jax-fem 요소 타입별 VTK cell type 매핑 테이블
- [ ] Transient: 타임스텝 결과를 전체 메모리에 보관 vs 디스크 스트리밍

---

## 5. Exporter Layer (`wht_exporters.py`)

### 5.1 `BaseExporter`

```python
class BaseExporter(ABC):
    @abstractmethod
    def export(self, data: WHTResultData, output_path: str) -> None: ...

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """output_path의 부모 디렉토리가 없으면 자동 생성."""
```

---

### 5.2 `VTKHDFExporter` (Primary, ParaView 5.11+)

- **의존성**: `h5py` (`pip install h5py`)
- **출력**: 단일 `.hdf` 파일 (바이너리, 압축 지원)
- **장점**: 수백 개의 `.vtu` 파일을 단일 파일로 대체, ParaView 6 animation 지원

#### HDF5 파일 스키마

```
<output.hdf>
└── VTKHDF/                          ← 최상위 그룹
    ├── Type          (attr)  = "UnstructuredGrid"   string
    ├── Version       (attr)  = [2, 0]               int64[2]
    │
    ├── Points        (N, 3)          float64   ← 정적 메시 (1회 저장)
    ├── Connectivity  (K,)            int64
    ├── Offsets       (M+1,)          int64
    ├── Types         (M,)            uint8
    ├── NumberOfPoints (1,)           int64     = [N]
    ├── NumberOfCells  (1,)           int64     = [M]
    │
    ├── Steps/
    │   ├── NSteps    (attr)  = T
    │   ├── Values    (T,)    float64  ← time / freq [Hz] / load_factor
    │   ├── PointOffsets (T,) int64    = [0, N, 2N, ..., (T-1)*N]
    │   ├── CellOffsets  (T,) int64    = [0, M, 2M, ..., (T-1)*M]
    │   └── PartOffsets  (T,) int64    = [0, 0, ..., 0]
    │                                    ↑ ParaView 6 필수. serial 데이터에서도 항상 0.
    │
    ├── PointData/
    │   └── <name>    (T*N, D)  float32  ← arr[t_idx*N : (t_idx+1)*N, :]
    │                                        = time_step t_idx의 노달 결과
    └── CellData/
        └── <name>    (T*M, D)  float32
```

> **ParaView에서 Animation 재생 방법**:
> 파일 열기 → Apply → View > Animation View → Play
> `Steps/Values`가 time axis로 사용됨 (modal이면 주파수 [Hz], transient이면 시간 [s])

#### 생성자 파라미터

```python
VTKHDFExporter(
    compression: str | None = "gzip",  # "gzip" | "lzf" | None
    compression_opts: int = 4,          # gzip level 1–9
    chunk_timesteps: int = 10,          # HDF5 chunk 단위 타임스텝 수
)
```

---

### 5.3 `VTUPVDExporter` (Legacy, meshio 미사용)

- **의존성**: Python 표준 라이브러리만 사용 (`xml.etree.ElementTree`)
- **출력**: `.pvd` (인덱스) + `_t0000.vtu`, `_t0001.vtu`, ... (타임스텝별)
- **용도**: 추가 의존성 없이 ParaView에서 결과 확인할 때 사용

```
<stem>.pvd
    → Collection of DataSet entries (timestep=t_val, file=stem_t0000.vtu, ...)

<stem>_t0000.vtu
    → VTKFile type="UnstructuredGrid"
        Points, Cells (connectivity/offsets/types)
        PointData: <name> for each point_data key at t=0
        CellData:  <name> for each cell_data key at t=0
```

---

### 5.4 `HWASCIIExporter` (Altair HyperView)

- **의존성**: 없음
- **출력**: Altair Generic ASCII 포맷 (`.ascii`)

#### 지원 결과 블록

| 분류 | 블록 이름 | `WHTResultData` 매핑 | 컬럼 |
|------|-----------|----------------------|------|
| Point | `Displacement` | `point_data["Displacement"]` (static) | UX, UY, UZ |
| Point | `Eigen`        | `point_data["Displacement"]` (modal)  | UX, UY, UZ |
| Point | `BucklingMode` | `point_data["BucklingMode"]`          | UX, UY, UZ |
| Cell  | `Stress`       | `cell_data["Stress"]`                 | S11, S22, S33, S12, S13, S23 |
| Cell  | `Strain`       | `cell_data["Strain"]`                 | E11, E22, E33, E12, E13, E23 |

> **구현 상세**: `SUPPORTED_POINT_BLOCKS = {"Displacement", "Eigen", "BucklingMode"}`,
> `SUPPORTED_CELL_BLOCKS = {"Stress", "Strain"}` 으로 분리하여 관리.
> 지원하지 않는 필드는 `WHTExportWarning`을 발생시키고 스킵한다.

#### 파일 구조 템플릿

```
$ALTAIR_ASCII_RESULT 1.0
$ANALYSIS_TYPE       modal
$SOLVER              JaxSSO 0.1.0
$CREATED_AT          2026-04-18T09:00:00Z
$UNITS               mm N
$NODES               441
$ELEMENTS            400

$TIME 1.23456789e+01  $ Frequency [Hz]

$RESULT_TYPE Eigen
$RESULT_LOCATION Node
$NODE_ID              UX              UY              UZ
       1    1.234567e-03    2.345678e-04    0.000000e+00
       ...

$END_TIME
```

---

## 6. Error Handling Strategy

```
솔버 출력
   │
   ▼
[Adapter._convert_*()]
   │  results dict에 필수 키 없음 → WHTValidationError (즉시 중단)
   │
   ▼
WHTResultData.__post_init__()
   │  shape 불일치 → WHTValidationError (즉시 중단)
   │
   ▼
[Adapter.validate()]
   │  transient: 단조성 위반 → WHTValidationError (즉시 중단)
   │  modal: 주파수 ≤ 0    → WHTValidationError (즉시 중단)
   │  buckling: 하중계수 ≤ 0 → WHTExportWarning (경고, 계속)
   │
   ▼
[Exporter.export()]
   │  미지원 point_data / cell_data 키 → WHTExportWarning (경고, 스킵 후 계속)
   │
   ▼
출력 파일
```

---

## 7. CLI Interface (`__main__.py`)

### 사용법

```bash
# 전체 포맷 출력 (기본값)
python -m wht_converter --input exam1_nf.py --output results/

# 특정 포맷만
python -m wht_converter --input exam1_nf.py --export vtkhdf --output results/

# 압축 옵션 변경
python -m wht_converter --input exam1_nf.py --export vtkhdf \
    --compression gzip --compression-level 6

# IR 검증만 (파일 미생성)
python -m wht_converter --input exam1_nf.py --dry-run

# 경고를 오류로 처리 (strict 모드)
python -m wht_converter --input exam1_nf.py --warn-errors
```

### 전체 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input`, `-i` | (필수) | 솔버 스크립트 경로 |
| `--export`, `-e` | `"all"` | `"all"` \| `"vtkhdf"` \| `"vtu"` \| `"hwascii"` |
| `--output`, `-o` | `"results/"` | 출력 디렉토리 |
| `--compression` | `"gzip"` | `"gzip"` \| `"lzf"` \| `"none"` |
| `--compression-level` | `4` | gzip 압축 수준 1–9 |
| `--chunk-timesteps` | `10` | VTKHDF HDF5 청크 단위 |
| `--dry-run` | `False` | IR 검증 후 파일 미생성 |
| `--warn-errors` | `False` | WHTExportWarning을 오류로 처리 |

### 솔버 스크립트 요구사항

CLI가 동적으로 import하여 호출할 함수를 스크립트에 정의해야 한다:

```python
# exam1_nf.py에 추가 필요 (현재 미구현 — Phase 4 TODO)
def get_wht_data() -> WHTResultData:
    """CLI --input 인터페이스. solver 실행 후 WHTResultData를 반환."""
    ...
    return adapter.convert(model, results, "modal", meta)
```

> **현재 상태**: `exam1_nf.py`는 `ParaViewExporter` 클래스로 `main()` 내부에서
> 직접 export를 수행하며, CLI를 통한 호출은 지원하지 않는다.

---

## 8. `exam1_nf.py` — `ParaViewExporter` 사용 패턴

`exam1_nf.py`에서 wht_converter를 사용하는 방식. JaxSSO 모델과 ModalSolver 결과를
wht_converter에 연결하는 인터페이스 코드.

```
exam1_nf.py 의존 관계
─────────────────────────────────────────────────────
ModalSolver.solve()
    │  vecs_full: (ndof, n_modes) = (n_nodes*6, n_modes)
    │  freqs:     (n_modes,)  [Hz]
    │
    ▼
ParaViewExporter._reshape_vecs()
    │  vecs_full.T.reshape(n_modes, n_nodes, 6)
    │  단, node_id 순서 정합: sorted(node_db.keys())[j] → vecs_full[nid*6:nid*6+6]
    │  → vecs_3d: (n_modes, n_nodes, 6)
    │
    ▼
ParaViewExporter._ModelProxy(node_db, elem_db)
    │  .nodes = node_db   {nid: [x,y,z]}
    │  .quads = elem_db   {eid: [n0,n1,n2,n3]}
    │
    ▼
JaxSSOAdapter.convert(proxy, {"vecs": vecs_3d, "freqs": freqs}, "modal", meta)
    │
    ▼
VTUPVDExporter / VTKHDFExporter
    → results/D날짜-시간/modal_result.pvd
    → results/D날짜-시간/modal_result.hdf
```

> **DOF 레이아웃 가정**: JaxSSO가 노드 `nid`에 대해 DOF를 `nid*6` ~ `nid*6+5` 위치에
> 배치한다고 가정. node_id가 0-based contiguous (0, 1, 2, ...)인 경우에만 성립.
> 비연속 ID라면 `_reshape_vecs()` 내의 명시적 루프가 정확한 매핑을 보장한다.

---

## 9. 구현 현황 (Phase 체크리스트)

### Phase 0 — 솔버 데이터 구조 조사
- [x] JaxSSO: `model.nodes`, `quads`, `beamcols`, `truss` 구조 확인
- [ ] jax-fem: `problem.mesh` 연결 배열 필드명·레이아웃 확인

### Phase 1 — Core Models & Utilities
- [x] `wht_models.py`: WHTMetadata, WHTResultData, WHTValidationError, WHTExportWarning
- [x] `wht_utils.py`: VTKCellType, to_vtk_csr, merge_csr, node_dict_to_array, remap_connectivity
- [x] `tests/test_models_and_utils.py`

### Phase 2 — Exporter Development
- [x] `BaseExporter` ABC
- [x] `VTKHDFExporter` (h5py, gzip/lzf 압축, 청킹, PartOffsets)
- [x] `VTUPVDExporter` (직접 XML, meshio 미사용)
- [x] `HWASCIIExporter` (5개 블록 타입)
- [x] `tests/test_exporters.py`

### Phase 3 — Adapter Development
- [x] `BaseAdapter` ABC + validate()
- [x] `JaxSSOAdapter` (Static / Modal / Buckling)
- [x] `JaxFEMAdapter` (Static / Transient / Modal, _extract_mesh 구현)
- [x] `tests/test_adapters.py`

### Phase 4 — Integration & UX
- [x] `exam1_nf.py`: `ParaViewExporter` 클래스 추가, VTU/PVD + VTKHDF 동시 출력
- [x] `wht_converter/__main__.py`: CLI 엔트리포인트 (import 버그 수정 완료)
- [ ] `exam1_nf.py`에 `get_wht_data()` 함수 추가 (CLI 연동)
- [ ] `tests/test_roundtrip.py`: 솔버 → 변환 → 파일 로드 → 원본 값 일치 확인
- [ ] `docs/USAGE.md`

---

## 10. Dependencies

| 패키지 | 용도 | 최소 버전 | 필수 여부 |
|--------|------|-----------|-----------|
| `numpy` | 수치 배열 처리 | 1.24+ | 필수 |
| `h5py` | VTKHDF HDF5 파일 읽기/쓰기 | 3.8+ | VTKHDFExporter 사용 시 |
| `jax` | JAX DeviceArray → NumPy 변환 | 0.4+ | JaxFEMAdapter 사용 시 |
| `pytest` | 단위 테스트 | 7.0+ | 개발 시 |
| ~~`meshio`~~ | ~~VTU/PVD 출력~~ | — | **미사용** (직접 XML 작성으로 대체) |

---

## 11. Key Benefits

| 항목 | 설명 |
|------|------|
| **Zero Redundancy** | 정적 메시 데이터는 VTKHDF에 1회만 저장, T 타임스텝의 결과만 반복 |
| **Single File** | 단일 `.hdf` 파일로 수백 개 VTU 대체 |
| **Library Agnostic** | 솔버 교체 시 어댑터만 교체, 익스포터 불변 |
| **Mixed Mesh Support** | Beam + Shell + Solid 혼합 요소를 CSR로 정확히 표현 |
| **Extensible** | `BaseAdapter` / `BaseExporter` ABC로 신규 솔버·포맷 독립 추가 |
| **Early Failure** | `__post_init__` + `validate()`로 조용한 버그 조기 차단 |
| **ParaView 6 Ready** | `PartOffsets` 포함, VTKHDF v2.0 스키마 완전 준수 |

---

## 12. Change Summary

| 항목 | v0.4 설계 | v0.5 (구현 동기화) |
|------|----------|-------------------|
| VTKHDF `PartOffsets` | 스키마에 없음 | **추가됨 — ParaView 6 필수** |
| `WHTResultData` 검증 | `assert` 사용 | **`WHTValidationError` 예외 사용** |
| `WHTMetadata` 검증 | 미문서화 | **`__post_init__` 검증 로직 추가** |
| `wht_utils.py` 함수 | `to_vtk_csr()`만 | **`merge_csr`, `node_dict_to_array`, `remap_connectivity` 추가** |
| `JaxSSOAdapter._extract_nodes` | 어댑터 내부 메서드 | **`node_dict_to_array()` (utils)로 분리** |
| `VTUPVDExporter` 구현 | meshio 또는 직접 | **직접 XML 작성으로 확정 (meshio 미사용)** |
| CLI 시그니처 | positional arg | **`--input/-i` 플래그로 변경** |
| CLI 추가 옵션 | 없음 | **`--dry-run`, `--warn-errors`, `--chunk-timesteps` 추가** |
| `HWASCIIExporter` 블록 | 단일 `SUPPORTED_BLOCKS` set | **`SUPPORTED_POINT_BLOCKS` / `SUPPORTED_CELL_BLOCKS` 분리** |
| 테스트 위치 | `tests/` (프로젝트 루트) | **`wht_converter/tests/` (패키지 내부)** |
| `__main__.py` imports | 절대 import | **상대 import (`from .wht_exporters`)로 수정** |
| 구현 현황 | Phase 0–4 모두 미완 | **Phase 0–3 완료, Phase 4 부분 완료** |

---

*This document reflects the actual implementation state as of 2026-04-18.*
