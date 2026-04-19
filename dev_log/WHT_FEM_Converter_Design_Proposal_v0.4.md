# [Design Proposal] WHT Universal FEM Result Converter

**Date**: 2026-04-18
**Version**: 0.4 (Ultra-Detailed)
**Status**: Pending Review
**Target**: Commercial-grade FEM Post-processing Engine
**Target Solvers**: JaxSSO (Static / Modal / Buckling), jax-fem (Static / Transient / Modal)
**Target Viewers**: ParaView 5.11+, Altair HyperView

---

## 1. Overview & Design Principles

JAX 기반 FEM 솔버(JaxSSO, jax-fem)와 산업용 후처리 도구(ParaView, HyperView) 사이의
명확하고 결합도 낮은 변환 인터페이스를 제공한다.

### 1.1 Design Principles

| 원칙 | 설명 |
|------|------|
| **Single Responsibility** | 데이터 컨테이너, 어댑터, 익스포터는 각각 독립 교체 가능 |
| **Open/Closed** | 새 솔버·포맷 추가 시 기존 코드 수정 없이 서브클래스만 작성 |
| **Explicit Validation** | 각 단계 경계에서 shape·단위·타입을 명시적으로 검사 |
| **Fail Fast** | 오류는 가능한 한 조기에 발생시켜 후처리 단계의 조용한 버그를 방지 |

### 1.2 Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ Solver      │     │ BaseAdapter      │     │ WHTResultData (IR)   │     │ BaseExporter │
│ (JaxSSO /   │────▶│ .convert()       │────▶│ • nodes              │────▶│ .export()    │
│  jax-fem)   │     │ .validate()      │     │ • connectivity/offset│     │              │
└─────────────┘     └──────────────────┘     │ • point_data / cell_ │     └──────┬───────┘
                                             │   data / field_data  │            │
                                             │ • time_values        │     ┌──────▼──────────────────┐
                                             │ • metadata           │     │ VTKHDFExporter (.hdf)   │
                                             └──────────────────────┘     │ VTUPVDExporter (.vtu)   │
                                                                          │ HWASCIIExporter (.ascii) │
                                                                          └─────────────────────────┘
```

---

## 2. Internal Architecture

### 2.1 Intermediate Representation (IR): `WHTMetadata`

결과 데이터와 분리된 메타정보 컨테이너. 익스포터가 포맷별 헤더를 독립적으로 작성할 수 있게 한다.

```python
@dataclass
class WHTMetadata:
    solver_name: str          # "JaxSSO" | "jax-fem"
    solver_version: str       # 예: "0.1.0"
    analysis_type: str        # "static" | "modal" | "transient" | "buckling"
    coordinate_system: str    # "cartesian" | "cylindrical"
    unit_length: str          # "m" | "mm" | "in"
    unit_force: str           # "N" | "kN" | "lbf"
    created_at: str           # ISO 8601 (예: "2026-04-18T09:00:00Z")
```

> ⚠️ **단위 불일치 방지**: mm vs m 혼용 시 HyperView에서 스케일 1000배 오차가 발생한다.
> 어댑터는 반드시 `unit_length`를 명시적으로 전달해야 하며, 기본값을 허용하지 않는다.

---

### 2.2 Intermediate Representation (IR): `WHTResultData`

모든 솔버 출력과 익스포터 입력의 **단일 교환 포맷**.

```python
@dataclass
class WHTResultData:
    # --- Geometry (CSR Flat Format) ---
    nodes: np.ndarray                        # (N, 3)   노드 좌표 [x, y, z]
    connectivity: np.ndarray                 # (K,)     플랫 연결 배열 (모든 셀 노드 인덱스)
    offsets: np.ndarray                      # (M+1,)   각 셀의 시작 인덱스
    cell_types: np.ndarray                   # (M,)     VTK 셀 타입 상수

    # --- Named Sets (Optional) ---
    node_sets: dict[str, np.ndarray]         # {"fixed": np.array([0,1,2])}
    element_sets: dict[str, np.ndarray]      # {"beam_group": np.array([5,6,7])}

    # --- Results ---
    point_data: dict[str, np.ndarray]        # {name: (T, N, D)}  노달 결과
    cell_data:  dict[str, np.ndarray]        # {name: (T, M, D)}  요소 결과
    field_data: dict[str, np.ndarray]        # {name: (T,)}       전역 스칼라

    # --- Time / Mode / Load Factor Axis ---
    time_values: np.ndarray                  # (T,)  분석 유형에 따라 의미가 다름
                                             #   static:    [0.0]
                                             #   transient: [t0, t1, ..., tT]
                                             #   modal:     [freq0, freq1, ...]  (Hz)
                                             #   buckling:  [λ0, λ1, ...]  (하중계수)

    # --- Metadata ---
    metadata: WHTMetadata

    def __post_init__(self):
        """생성 즉시 기본적인 shape 일관성을 검사한다."""
        N, M = self.nodes.shape[0], len(self.offsets) - 1
        T = len(self.time_values)

        assert self.nodes.ndim == 2 and self.nodes.shape[1] == 3, \
            f"nodes must be (N, 3), got {self.nodes.shape}"
        assert len(self.offsets) == M + 1, \
            f"offsets length must be M+1={M+1}, got {len(self.offsets)}"
        assert len(self.cell_types) == M, \
            f"cell_types length must be M={M}, got {len(self.cell_types)}"

        for name, arr in self.point_data.items():
            assert arr.shape[0] == T and arr.shape[1] == N, \
                f"point_data['{name}'] must be (T={T}, N={N}, D), got {arr.shape}"
        for name, arr in self.cell_data.items():
            assert arr.shape[0] == T and arr.shape[1] == M, \
                f"cell_data['{name}'] must be (T={T}, M={M}, D), got {arr.shape}"
```

#### CSR Flat Format 상세

혼합 요소(Mixed-Dimensional) 메시를 지원하기 위해 VTK와 동일한 플랫+오프셋 방식을 사용한다.

```
# 예: Beam(2노드) 1개 + Quad(4노드) 1개 + Hexa(8노드) 1개 혼합 메시
connectivity = [0, 1,   2, 3, 4, 5,   6, 7, 8, 9, 10, 11, 12, 13]
offsets      = [0, 2,   6,             14]
cell_types   = [3,      9,             12]
               └─Beam   └─Quad          └─Hexa

# VTK 타입 상수 (자주 쓰이는 것)
VTK_LINE         =  3   # 2-node beam/truss
VTK_TRIANGLE     =  5   # 3-node triangle
VTK_QUAD         =  9   # 4-node quad (shell)
VTK_TETRA        = 10   # 4-node tetrahedron
VTK_HEXAHEDRON   = 12   # 8-node hexahedron
VTK_WEDGE        = 13   # 6-node wedge/prism
```

#### 유틸리티: `to_vtk_csr()`

기존 `(M, V)` 형식의 단일-요소-타입 배열을 CSR 포맷으로 변환하는 헬퍼.

```python
def to_vtk_csr(
    elements: np.ndarray,   # (M, V) — 모든 요소가 동일한 노드 수 V를 가진다고 가정
    vtk_type: int,          # VTK cell type 상수
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        connectivity : (M*V,)
        offsets      : (M+1,)
        cell_types   : (M,)
    """
    M, V = elements.shape
    connectivity = elements.flatten()
    offsets      = np.arange(0, (M + 1) * V, V, dtype=np.int64)
    cell_types   = np.full(M, vtk_type, dtype=np.uint8)
    return connectivity, offsets, cell_types
```

---

## 3. Adapter Layer

### 3.1 Abstract Base Class

```python
from abc import ABC, abstractmethod

class BaseAdapter(ABC):

    @abstractmethod
    def convert(self, *args, **kwargs) -> WHTResultData:
        """솔버 네이티브 데이터를 WHTResultData (IR)로 변환한다."""
        ...

    def validate(self, data: WHTResultData) -> None:
        """
        __post_init__ 이후 추가 검증.
        - time_values가 단조 증가하는지 (transient)
        - modal 해석 시 freqs > 0인지
        - buckling 해석 시 load_factors 정렬 순서
        """
        if data.metadata.analysis_type == "transient":
            assert np.all(np.diff(data.time_values) > 0), \
                "time_values must be monotonically increasing for transient analysis"
        if data.metadata.analysis_type == "modal":
            assert np.all(data.time_values > 0), \
                "Modal frequencies must be positive"
```

---

### 3.2 JaxSSOAdapter

#### 확인된 데이터 구조 (from 소스 조사)

| 항목 | 필드 | 비고 |
|------|------|------|
| 노드 좌표 | `model.nodes` | `dict {node_id: [x, y, z]}` — ID 정렬 필요 |
| Quad 요소 | `model.quads` | `dict {elem_id: [n0, n1, n2, n3]}` |
| Beam/Col 요소 | `model.beamcols` | `dict {elem_id: [n0, n1]}` |
| Truss 요소 | `model.truss` | `dict {elem_id: [n0, n1]}` |

> 노드/요소 ID가 연속적이지 않을 수 있으므로, `sorted()` 후 0-based 인덱스로 재매핑한다.

#### 구현

```python
class JaxSSOAdapter(BaseAdapter):

    def convert(
        self,
        model,
        results: dict,
        analysis_type: str,  # "static" | "modal" | "buckling"
        metadata: WHTMetadata,
    ) -> WHTResultData:
        # --- Geometry 공통 처리 ---
        nodes, node_id_map = self._extract_nodes(model)
        connectivity, offsets, cell_types = self._extract_mesh(model, node_id_map)

        # --- 분석 유형별 결과 처리 ---
        if analysis_type == "static":
            point_data, cell_data, field_data, time_values = \
                self._convert_static(results, node_id_map)
        elif analysis_type == "modal":
            point_data, cell_data, field_data, time_values = \
                self._convert_modal(results, node_id_map)
        elif analysis_type == "buckling":
            point_data, cell_data, field_data, time_values = \
                self._convert_buckling(results, node_id_map)
        else:
            raise WHTValidationError(f"Unsupported analysis_type: '{analysis_type}'")

        data = WHTResultData(
            nodes=nodes,
            connectivity=connectivity, offsets=offsets, cell_types=cell_types,
            node_sets={}, element_sets={},
            point_data=point_data, cell_data=cell_data, field_data=field_data,
            time_values=time_values, metadata=metadata,
        )
        self.validate(data)
        return data

    def _extract_nodes(self, model) -> tuple[np.ndarray, dict]:
        """model.nodes (dict) → (N,3) ndarray + id→index 매핑 반환"""
        sorted_ids = sorted(model.nodes.keys())
        node_id_map = {nid: i for i, nid in enumerate(sorted_ids)}
        nodes = np.array([model.nodes[nid] for nid in sorted_ids], dtype=np.float64)
        return nodes, node_id_map

    def _extract_mesh(self, model, node_id_map: dict):
        """
        model.quads / beamcols / truss를 순회하여
        혼합 CSR 메시를 구성한다.
        """
        all_conn, all_offsets, all_types = [], [0], []
        offset = 0

        elem_groups = [
            (model.quads,     4, 9),   # VTK_QUAD
            (model.beamcols,  2, 3),   # VTK_LINE
            (model.truss,     2, 3),   # VTK_LINE
        ]
        for elem_dict, n_nodes, vtk_type in elem_groups:
            for eid in sorted(elem_dict.keys()):
                local_nodes = [node_id_map[nid] for nid in elem_dict[eid]]
                all_conn.extend(local_nodes)
                offset += n_nodes
                all_offsets.append(offset)
                all_types.append(vtk_type)

        return (
            np.array(all_conn,    dtype=np.int64),
            np.array(all_offsets, dtype=np.int64),
            np.array(all_types,   dtype=np.uint8),
        )

    def _convert_static(self, results, node_id_map):
        u = np.asarray(results["u"])          # (N, D)
        # T=1 차원 추가 → (1, N, D)
        point_data  = {"Displacement": u[np.newaxis, :, :]}
        cell_data   = {}
        field_data  = {}
        time_values = np.array([0.0])
        return point_data, cell_data, field_data, time_values

    def _convert_modal(self, results, node_id_map):
        vecs  = np.asarray(results["vecs"])   # (n_modes, N, D)
        freqs = np.asarray(results["freqs"])  # (n_modes,)  단위: Hz 가정
        point_data  = {"Displacement": vecs}
        cell_data   = {}
        field_data  = {}
        time_values = freqs
        return point_data, cell_data, field_data, time_values

    def _convert_buckling(self, results, node_id_map):
        modes        = np.asarray(results["modes"])        # (n_modes, N, D)
        load_factors = np.asarray(results["load_factors"]) # (n_modes,)
        # 양의 하중계수만 유효 → 경고 후 필터링
        if np.any(load_factors <= 0):
            import warnings
            warnings.warn(
                "Negative or zero load factors detected; these may indicate "
                "non-physical buckling modes.",
                WHTExportWarning,
            )
        point_data  = {"BucklingMode": modes}
        cell_data   = {}
        field_data  = {"LoadFactor": load_factors}
        time_values = load_factors
        return point_data, cell_data, field_data, time_values
```

#### 잔여 조사 항목

- [ ] `model.quads` 노드 순서가 VTK Quad 와인딩 규약(CCW)과 일치하는지 확인
- [ ] Modal 고유진동수 단위: rad/s vs Hz (현재 Hz로 가정)
- [ ] Buckling `load_factors` 정렬 순서: 오름차순 가정 여부
- [ ] 응력·반력 결과의 후처리 출력 여부 (직접 제공 vs 별도 계산)

---

### 3.3 JaxFEMAdapter

#### 확인된 데이터 구조 (from 소스 조사)

| 항목 | 필드 | 비고 |
|------|------|------|
| 노드 좌표 | `problem.mesh.node_coords` | JAX `DeviceArray` → `np.asarray()` 필요 |
| 요소 연결 | *(조사 필요)* | 플랫 vs 2D 배열 미확인 |
| VTK 타입 | *(조사 필요)* | jax-fem 요소 타입 → VTK 상수 매핑 미확인 |

#### 구현

```python
class JaxFEMAdapter(BaseAdapter):

    def convert(
        self,
        problem,
        results: dict,
        analysis_type: str,  # "static" | "transient" | "modal"
        metadata: WHTMetadata,
    ) -> WHTResultData:
        nodes        = np.asarray(problem.mesh.node_coords)  # DeviceArray → NumPy
        connectivity, offsets, cell_types = self._extract_mesh(problem.mesh)

        if analysis_type == "static":
            point_data, cell_data, field_data, time_values = \
                self._convert_static(results)
        elif analysis_type == "transient":
            point_data, cell_data, field_data, time_values = \
                self._convert_transient(results)
        elif analysis_type == "modal":
            point_data, cell_data, field_data, time_values = \
                self._convert_modal(results)
        else:
            raise WHTValidationError(f"Unsupported analysis_type: '{analysis_type}'")

        data = WHTResultData(
            nodes=nodes,
            connectivity=connectivity, offsets=offsets, cell_types=cell_types,
            node_sets={}, element_sets={},
            point_data=point_data, cell_data=cell_data, field_data=field_data,
            time_values=time_values, metadata=metadata,
        )
        self.validate(data)
        return data

    def _extract_mesh(self, mesh):
        """
        ⚠️ TODO: jax-fem 요소 연결 배열 구조 확인 후 구현
        현재는 단일 요소 타입 가정 (to_vtk_csr 활용)
        """
        raise NotImplementedError(
            "jax-fem mesh extraction requires investigation. "
            "See checklist in Section 3.3."
        )

    def _convert_static(self, results):
        u = np.asarray(results["u"])          # (N, D)
        return {"Displacement": u[np.newaxis]}, {}, {}, np.array([0.0])

    def _convert_transient(self, results):
        u      = np.asarray(results["u"])     # (T, N, D)
        t_vals = np.asarray(results["t"])     # (T,)
        return {"Displacement": u}, {}, {}, t_vals

    def _convert_modal(self, results):
        eigvecs = np.asarray(results["eigvecs"])  # (n_modes, N, D)
        eigvals = np.asarray(results["eigvals"])  # (n_modes,)
        # rad/s → Hz 변환 (가정)
        freqs = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
        return {"Displacement": eigvecs}, {}, {}, freqs
```

#### 잔여 조사 항목

- [ ] `problem.mesh` 요소 연결 배열 필드명 및 레이아웃 (플랫 vs 2D)
- [ ] jax-fem 요소 타입 → VTK cell type 매핑 테이블
- [ ] Transient: 중간 타임스텝 결과를 메모리에 보관하는 방식 vs 디스크 스트리밍
- [ ] Modal: `eigvals` 단위 (rad²/s² 가정 여부 확인)
- [ ] 응력·변형률 결과의 후처리 출력 여부

---

## 4. Exporter Layer

### 4.1 Abstract Base Class

```python
class BaseExporter(ABC):
    @abstractmethod
    def export(self, data: WHTResultData, output_path: str) -> None:
        ...
```

---

### 4.2 VTU/PVD Exporter

- **용도**: 레거시 ParaView 지원 및 소규모 결과 시각적 검증용
- **라이브러리**: `meshio` 또는 직접 XML 작성
- **상태**: 보조 포맷 (기본 출력은 VTKHDF)

---

### 4.3 VTKHDF Exporter (Primary)

- **용도**: ParaView 5.11+ 단일 HDF5 파일 출력 (`.hdf`)
- **라이브러리**: `h5py`

#### HDF5 파일 스키마

```
<output.hdf>
└── VTKHDF/
    ├── Type              (attr: "UnstructuredGrid")
    ├── Version           (attr: [2, 0])
    │
    ├── Points            (N, 3)          float64  ← 정적, 1회 저장
    ├── Connectivity      (K,)            int64    ← 정적, 1회 저장
    ├── Offsets           (M+1,)          int64    ← 정적, 1회 저장
    ├── Types             (M,)            uint8    ← 정적, 1회 저장
    ├── NumberOfPoints    (1,)            int64
    ├── NumberOfCells     (1,)            int64
    │
    ├── Steps/
    │   ├── NSteps        (attr: T)
    │   ├── Values        (T,)            float64  ← time / freq / load factor
    │   ├── PointOffsets  (T,)            int64
    │   └── CellOffsets   (T,)            int64
    │
    ├── PointData/
    │   └── Displacement  (T*N, D)        float32  ← 타임스텝별 청킹
    │
    └── CellData/
        ├── Stress        (T*M, 6)        float32
        └── Strain        (T*M, 6)        float32
```

#### 구현

```python
class VTKHDFExporter(BaseExporter):

    def export(
        self,
        data: WHTResultData,
        output_path: str,
        compression: str = "gzip",    # "gzip" | "lzf" | None
        compression_opts: int = 4,    # gzip 레벨 (1~9)
        chunk_timesteps: int = 10,    # 청킹 단위 타임스텝 수
    ) -> None:
        import h5py
        T, N = len(data.time_values), data.nodes.shape[0]
        M    = len(data.offsets) - 1
        comp_kwargs = {"compression": compression, "compression_opts": compression_opts} \
                      if compression else {}

        with h5py.File(output_path, "w") as f:
            grp = f.create_group("VTKHDF")
            grp.attrs["Type"]    = "UnstructuredGrid"
            grp.attrs["Version"] = [2, 0]

            # --- Static Geometry (1회 저장) ---
            grp.create_dataset("Points",       data=data.nodes.astype(np.float64))
            grp.create_dataset("Connectivity", data=data.connectivity.astype(np.int64))
            grp.create_dataset("Offsets",      data=data.offsets.astype(np.int64))
            grp.create_dataset("Types",        data=data.cell_types.astype(np.uint8))
            grp.create_dataset("NumberOfPoints", data=np.array([N], dtype=np.int64))
            grp.create_dataset("NumberOfCells",  data=np.array([M], dtype=np.int64))

            # --- Steps ---
            steps = grp.create_group("Steps")
            steps.attrs["NSteps"] = T
            steps.create_dataset("Values", data=data.time_values.astype(np.float64))
            steps.create_dataset("PointOffsets",
                                 data=(np.arange(T) * N).astype(np.int64))
            steps.create_dataset("CellOffsets",
                                 data=(np.arange(T) * M).astype(np.int64))

            # --- PointData ---
            pd_grp = grp.create_group("PointData")
            for name, arr in data.point_data.items():
                # arr: (T, N, D) → (T*N, D)
                flat = arr.reshape(T * N, -1).astype(np.float32)
                chunk = (min(chunk_timesteps, T) * N, flat.shape[1])
                pd_grp.create_dataset(name, data=flat, chunks=chunk, **comp_kwargs)

            # --- CellData ---
            cd_grp = grp.create_group("CellData")
            for name, arr in data.cell_data.items():
                flat = arr.reshape(T * M, -1).astype(np.float32)
                chunk = (min(chunk_timesteps, T) * M, flat.shape[1])
                cd_grp.create_dataset(name, data=flat, chunks=chunk, **comp_kwargs)
```

---

### 4.4 HWASCII Exporter (HyperView)

- **용도**: Altair HyperView 결과 오버레이 (`.ascii`)
- **포맷**: Altair Generic ASCII

#### 지원 결과 블록

| 블록 타입      | 결과 종류                      | `WHTResultData` 매핑        |
|----------------|--------------------------------|-----------------------------|
| `Displacement` | 노드 변위 (UX, UY, UZ)         | `point_data["Displacement"]` |
| `Stress`       | 요소 응력 텐서 (S11~S33)        | `cell_data["Stress"]`       |
| `Strain`       | 요소 변형률 텐서 (E11~E33)      | `cell_data["Strain"]`       |
| `Eigen`        | 고유벡터 (Modal 변위)           | `point_data["Displacement"]` (analysis_type=modal) |
| `Buckling`     | 좌굴 모드 변위 + 하중계수        | `point_data["BucklingMode"]`, `field_data["LoadFactor"]` |

#### HWASCII 파일 구조 (템플릿)

```
$ALTAIR_ASCII_RESULT 1.0
$ANALYSIS_TYPE       {analysis_type}
$UNITS               {unit_length} {unit_force}
$NODES               {N}
$ELEMENTS            {M}

# --- 타임스텝/모드 반복 ---
$TIME {t_val}

$RESULT_TYPE Displacement
$RESULT_LOCATION Node
$NODE_ID  UX         UY         UZ
{node_id:>8d}  {ux:>12.6e}  {uy:>12.6e}  {uz:>12.6e}
...

$RESULT_TYPE Stress
$RESULT_LOCATION Element
$ELEM_ID  S11  S22  S33  S12  S13  S23
...

$END_TIME
```

#### 구현

```python
class HWASCIIExporter(BaseExporter):

    SUPPORTED_BLOCKS = {"Displacement", "Stress", "Strain", "Eigen", "Buckling"}

    def export(self, data: WHTResultData, output_path: str) -> None:
        import warnings
        meta = data.metadata
        T    = len(data.time_values)
        N    = data.nodes.shape[0]
        M    = len(data.offsets) - 1

        with open(output_path, "w") as f:
            # --- 파일 헤더 ---
            f.write(f"$ALTAIR_ASCII_RESULT 1.0\n")
            f.write(f"$ANALYSIS_TYPE       {meta.analysis_type}\n")
            f.write(f"$SOLVER              {meta.solver_name} {meta.solver_version}\n")
            f.write(f"$UNITS               {meta.unit_length} {meta.unit_force}\n")
            f.write(f"$NODES               {N}\n")
            f.write(f"$ELEMENTS            {M}\n\n")

            # --- 타임스텝/모드별 반복 ---
            for t_idx in range(T):
                t_val = data.time_values[t_idx]
                f.write(f"$TIME {t_val:.6e}\n\n")

                # PointData 블록
                for name, arr in data.point_data.items():
                    if name not in self.SUPPORTED_BLOCKS and \
                       not any(name.startswith(b) for b in self.SUPPORTED_BLOCKS):
                        warnings.warn(
                            f"point_data['{name}'] is not a supported block; skipping.",
                            WHTExportWarning,
                        )
                        continue
                    result_type = "Eigen" \
                        if meta.analysis_type == "modal" else name
                    f.write(f"$RESULT_TYPE {result_type}\n")
                    f.write(f"$RESULT_LOCATION Node\n")
                    step_data = arr[t_idx]  # (N, D)
                    for nid in range(N):
                        vals = "  ".join(f"{v:>12.6e}" for v in step_data[nid])
                        f.write(f"{nid+1:>8d}  {vals}\n")
                    f.write("\n")

                # CellData 블록
                for name, arr in data.cell_data.items():
                    if name not in self.SUPPORTED_BLOCKS:
                        warnings.warn(
                            f"cell_data['{name}'] is not a supported block; skipping.",
                            WHTExportWarning,
                        )
                        continue
                    f.write(f"$RESULT_TYPE {name}\n")
                    f.write(f"$RESULT_LOCATION Element\n")
                    step_data = arr[t_idx]  # (M, D)
                    for eid in range(M):
                        vals = "  ".join(f"{v:>12.6e}" for v in step_data[eid])
                        f.write(f"{eid+1:>8d}  {vals}\n")
                    f.write("\n")

                f.write("$END_TIME\n\n")
```

---

## 5. Error Handling Strategy

```
솔버 출력
   │
   ▼
[Adapter.convert()]  ──── WHTValidationError → 즉시 중단 (변환 실패)
   │
   ▼
WHTResultData.__post_init__()  ──── WHTValidationError → 즉시 중단 (shape 불일치)
   │
   ▼
[Adapter.validate()]  ──── WHTValidationError → 즉시 중단 (단조성, 부호 규약)
   │
   ▼
[Exporter.export()]  ──── WHTExportWarning → 경고 후 계속 (미지원 필드 스킵)
   │
   ▼
출력 파일
```

```python
class WHTValidationError(Exception):
    """데이터 구조·shape·단위 오류. 변환을 즉시 중단한다."""
    pass

class WHTExportWarning(UserWarning):
    """지원되지 않는 필드 스킵 등. 경고를 발생시키고 계속 진행한다."""
    pass
```

---

## 6. CLI Interface

```bash
# 모든 포맷 동시 출력
python -m wht_converter exam1_nf.py --export all --output results/

# 특정 포맷만 출력
python -m wht_converter exam1_nf.py --export vtkhdf --output results/

# 압축 옵션 지정
python -m wht_converter exam1_nf.py --export vtkhdf --compression gzip --compression-level 6
```

```python
# wht_converter/__main__.py
import argparse

EXPORTERS = {
    "vtkhdf": VTKHDFExporter,
    "vtu":    VTUPVDExporter,
    "hwascii": HWASCIIExporter,
    "all":    [VTKHDFExporter, VTUPVDExporter, HWASCIIExporter],
}

def main():
    parser = argparse.ArgumentParser(description="WHT FEM Result Converter")
    parser.add_argument("script",       help="Solver script path (e.g. exam1_nf.py)")
    parser.add_argument("--export",     choices=list(EXPORTERS.keys()), default="all")
    parser.add_argument("--output",     default="results/")
    parser.add_argument("--compression",default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--compression-level", type=int, default=4)
    args = parser.parse_args()
    # ... (솔버 실행 → 어댑터 → 익스포터 파이프라인 호출)
```

---

## 7. Phased Implementation Plan

### Phase 0 — 솔버 데이터 구조 조사 (선행 필수)

| 산출물 | 내용 |
|--------|------|
| `docs/jaxsso_data_survey.md` | `model.nodes`, `quads`, `beamcols`, `truss` 구조 검증 결과 |
| `docs/jaxfem_data_survey.md` | `problem.mesh` 연결 배열 레이아웃, VTK 타입 매핑 결과 |

---

### Phase 1 — Core Models & Utilities

- [ ] `wht_models.py`
  - `WHTMetadata` dataclass
  - `WHTResultData` dataclass (`__post_init__` 검증 포함)
  - `WHTValidationError`, `WHTExportWarning`
- [ ] `wht_utils.py`
  - `to_vtk_csr(elements, vtk_type)` 유틸리티 함수

---

### Phase 2 — Exporter Development

- [ ] `wht_exporters.py`
  - `BaseExporter` ABC
  - `VTKHDFExporter` (h5py 기반, 청킹/압축 포함)
  - `VTUPVDExporter` (meshio 기반)
  - `HWASCIIExporter` (5개 블록 타입 지원)
- [ ] `tests/test_exporters.py`
  - VTKHDF: h5py로 파일 로드 후 dataset 경로·shape·값 검증
  - HWASCII: 파일 파싱 후 `$RESULT_TYPE` 블록 수·값 검증

---

### Phase 3 — Adapter Development

- [ ] `wht_adapters.py`
  - `BaseAdapter` ABC
  - `JaxSSOAdapter` (Static / Modal / Buckling)
  - `JaxFEMAdapter` (Static / Transient / Modal)
- [ ] `tests/test_adapters.py`
  - 각 솔버·분석 유형별 변환 결과 shape 및 값 검증

---

### Phase 4 — Integration & UX

- [ ] `exam1_nf.py` 리팩터링: 기존 직접 출력 코드를 어댑터+익스포터 파이프라인으로 교체
- [ ] `wht_converter/__main__.py`: CLI 엔트리포인트 구현
- [ ] `tests/test_roundtrip.py`: 솔버 → 변환 → 파일 로드 → 원본 값 일치 확인
- [ ] `docs/USAGE.md`: 사용 예제 및 CLI 레퍼런스

---

### 7.1 Test Directory Structure

```
tests/
├── test_adapters.py
│   ├── test_jaxsso_static
│   ├── test_jaxsso_modal
│   ├── test_jaxsso_buckling
│   ├── test_jaxfem_static
│   ├── test_jaxfem_transient
│   └── test_jaxfem_modal
├── test_exporters.py
│   ├── test_vtkhdf_schema
│   ├── test_vtkhdf_compression
│   ├── test_hwascii_blocks
│   └── test_vtu_roundtrip
├── test_roundtrip.py
└── fixtures/
    ├── jaxsso_static_frame/       # 2D 프레임, Static
    ├── jaxsso_modal_truss/        # 3D 트러스, Modal
    ├── jaxsso_buckling_column/    # 기둥, Buckling
    ├── jaxfem_static_cube/        # 3D 고체, Static
    ├── jaxfem_transient_beam/     # 빔, Transient (10 스텝)
    └── jaxfem_modal_plate/        # 플레이트, Modal
```

---

## 8. Dependencies

| 패키지 | 용도 | 최소 버전 |
|--------|------|-----------|
| `numpy` | 수치 배열 처리 | 1.24+ |
| `h5py` | VTKHDF HDF5 파일 읽기/쓰기 | 3.8+ |
| `meshio` | VTU/PVD 출력 (선택) | 5.3+ |
| `jax` | JAX DeviceArray → NumPy 변환 | 0.4+ |
| `pytest` | 단위 테스트 | 7.0+ |

---

## 9. Key Benefits

| 항목 | 설명 |
|------|------|
| **Zero Redundancy** | 메시 데이터는 VTKHDF에 1회만 저장 |
| **Single File** | 단일 `.hdf` 파일로 수백 개 VTU 대체 |
| **Library Agnostic** | 솔버 교체 시 어댑터만 교체, 익스포터 불변 |
| **Mixed Mesh Support** | Beam + Shell + Solid 혼합 요소를 CSR로 정확히 표현 |
| **Extensible** | `BaseAdapter` / `BaseExporter` ABC로 신규 솔버·포맷 독립 추가 |
| **Early Failure** | `__post_init__` + `validate()`로 조용한 버그 조기 차단 |
| **CLI Ready** | `--export all`로 모든 포맷 일괄 출력 |

---

## 10. Change Summary (v0.1 → v0.4)

| 항목 | v0.1 | v0.2 | v0.3 | **v0.4** |
|------|------|------|------|----------|
| 메타데이터 | 없음 | `WHTMetadata` | 유지 | 유지 |
| 요소 연결 | `(M,V)` | `connectivity+offsets` | 유지 | 유지 + `to_vtk_csr()` 헬퍼 추가 |
| IR 검증 | 없음 | 없음 | 없음 | **`__post_init__` 자동 검증** |
| JaxSSO 어댑터 | 함수 | ABC | 3종 분기 | **`_extract_nodes/mesh` 상세 구현, 와인딩 경고** |
| jax-fem 어댑터 | 함수 | ABC | 3종 분기 | **`DeviceArray→NumPy`, `_extract_mesh` TODO 명시** |
| VTKHDF 스키마 | 미명시 | 옵션만 | 유지 | **HDF5 경로·dtype 전체 명시, 청킹 구현** |
| HWASCII 포맷 | 미명시 | 5종 블록 | 유지 | **`$RESULT_TYPE` 블록 템플릿 및 구현 추가** |
| 오류 처리 | 없음 | 2종 예외 | 유지 | **흐름도 + 예외 클래스 정의** |
| CLI | 없음 | 없음 | 없음 | **`--export all` CLI 추가** |
| 구현 계획 | 4단계 | 6단계 | 7단계 | **Phase 0~4 (산출물·테스트 명확화)** |
| 의존성 목록 | 없음 | 없음 | 없음 | **5개 패키지 최소 버전 명시** |

---

*Please review this proposal and provide your feedback or approval.*
