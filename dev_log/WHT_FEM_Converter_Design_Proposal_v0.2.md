# [Design Proposal] WHT Universal FEM Result Converter

**Date**: 2026-04-18
**Version**: 0.2 (Revised)
**Status**: Pending Review
**Target Solvers**: JaxSSO, jax-fem
**Target Viewers**: ParaView 5.11+, Altair HyperView

---

## 1. Overview

JAX 기반 FEM 솔버(JaxSSO, jax-fem)와 산업용 후처리 도구(ParaView, HyperView) 사이의
명확하고 결합도 낮은 변환 인터페이스를 제공한다.

컨버터는 HDF5 바이너리(VTKHDF) 및 Altair ASCII 명세의 복잡성을 추상화하며,
향후 솔버 및 출력 포맷 추가를 위한 확장 가능한 구조를 목표로 한다.

### 1.1 Design Principles

- **Single Responsibility**: 데이터 컨테이너, 어댑터, 익스포터는 각각 독립적으로 교체 가능해야 한다.
- **Open/Closed**: 새로운 솔버나 출력 포맷 추가 시 기존 코드 수정 없이 확장 가능해야 한다.
- **Explicit Validation**: 데이터 흐름의 각 단계에서 입력/출력의 유효성을 검사한다.

---

## 2. Technical Architecture

### 2.1 Metadata Container (`WHTMetadata`) ← NEW

결과 데이터와 분리된 메타정보를 보관한다.

```python
@dataclass
class WHTMetadata:
    solver_name: str          # "JaxSSO" | "jax-fem"
    solver_version: str
    analysis_type: str        # "static" | "modal" | "transient"
    coordinate_system: str    # "cartesian" | "cylindrical"
    unit_length: str          # "m" | "mm" | "in"
    unit_force: str           # "N" | "kN" | "lbf"
    created_at: str           # ISO 8601
```

> **Rationale**: 메타데이터를 수치 데이터와 분리하면 익스포터가 포맷별 헤더를
> 독립적으로 작성할 수 있고, 단위 불일치로 인한 버그를 조기에 차단할 수 있다.
> (예: mm vs m 혼용 시 HyperView에서 스케일 1000배 오차 발생 방지)

---

### 2.2 Standard Data Interface (`WHTResultData`)

A centralized hub for data exchange. It uses JAX/NumPy to handle large numerical arrays efficiently.

```python
@dataclass
class WHTResultData:
    # --- Geometry ---
    nodes: np.ndarray           # (N, 3)  노드 좌표
    connectivity: np.ndarray    # (K,)    플랫 연결 배열 (VTK CSR 방식)  ← CHANGED
    offsets: np.ndarray         # (M+1,)  각 요소의 시작 인덱스           ← NEW
    cell_types: np.ndarray      # (M,)    VTK 요소 타입 ID

    # --- Named Sets (Optional) ---                                        ← NEW
    node_sets: dict[str, np.ndarray]     # {"support": [0, 1, 2, ...]}
    element_sets: dict[str, np.ndarray]  # {"beam_group": [10, 11, ...]}

    # --- Results ---
    point_data: dict[str, np.ndarray]    # {name: (T, N, D)}  노달 결과
    cell_data:  dict[str, np.ndarray]    # {name: (T, M, D)}  요소 결과
    field_data: dict[str, np.ndarray]    # {name: (T,)}       전역 스칼라  ← NEW
    time_values: np.ndarray              # (T,) Time / Frequency / Mode ID

    # --- Metadata ---
    metadata: WHTMetadata                # 섹션 2.1 참조                   ← NEW
```

#### ⚠️ Breaking Change: `elements: (M, V)` → `connectivity + offsets`

원안의 `(M, V)` 형상은 모든 요소가 동일한 노드 수를 가진다고 가정한다.
실제 구조 해석 모델은 Hex + Tet + Wedge 등 **혼합 요소(Mixed Mesh)** 가 일반적이며,
V가 요소마다 다르므로 VTK 표준과 동일하게 **플랫 배열 + 오프셋** 방식으로 변경한다.

```
# 예: Tri(3노드) 1개 + Quad(4노드) 1개인 혼합 메시
connectivity = [0, 1, 2,  0, 2, 3, 4]
offsets      = [0, 3, 7]
```

---

### 2.3 Adapter Layer

솔버 네이티브 데이터를 `WHTResultData`로 변환하는 브릿지 레이어.

#### Abstract Base Class ← NEW

```python
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    def convert(self, *args, **kwargs) -> WHTResultData:
        """솔버 네이티브 데이터를 WHTResultData로 변환한다."""
        ...

    def validate(self, data: WHTResultData) -> None:
        """변환 결과의 형상(shape) 일관성을 검사한다."""
        N = data.nodes.shape[0]
        for name, arr in data.point_data.items():
            assert arr.shape[1] == N, f"point_data['{name}'] shape mismatch"
        # ... (추가 검증 로직)
```

> **Rationale**: `BaseAdapter` ABC를 정의하면 새로운 솔버(OpenSees, FEniCS 등)를
> 추가할 때 기존 코드를 수정하지 않고 서브클래스만 작성하면 된다.

#### Implementations

```python
class JaxSSOAdapter(BaseAdapter):
    def convert(self, model, vecs, freqs) -> WHTResultData:
        ...

class JaxFEMAdapter(BaseAdapter):
    def convert(self, problem_mesh, u, steps) -> WHTResultData:
        ...
```

---

### 2.4 Exporter Layer

`WHTResultData`를 각 포맷 파일로 출력하는 레이어.

#### Abstract Base Class ← NEW

```python
class BaseExporter(ABC):
    @abstractmethod
    def export(self, data: WHTResultData, output_path: str) -> None:
        ...
```

#### VTU/PVD Exporter

- **용도**: ParaView 레거시 지원, 소규모 결과 검증용
- **상태**: 유지 (보조 포맷)

#### VTKHDF Exporter

- **용도**: ParaView 5.11+ 고성능 단일 파일 출력 (`.hdf`)
- **추가 옵션** ← NEW:

```python
class VTKHDFExporter(BaseExporter):
    def export(
        self,
        data: WHTResultData,
        output_path: str,
        compression: str = "gzip",   # "gzip" | "lzf" | None
        chunk_timesteps: int = 10,   # 대용량 데이터 청킹 단위
    ) -> None:
        ...
```

#### HWASCII Exporter (HyperView)

지원 결과 블록을 명시한다 ← NEW:

| 블록 타입      | 결과 종류               | WHTResultData 매핑 |
|----------------|-------------------------|--------------------|
| `Displacement` | 노드 변위 (UX, UY, UZ)  | `point_data`       |
| `Stress`       | 요소 응력 텐서          | `cell_data`        |
| `Strain`       | 요소 변형률 텐서        | `cell_data`        |
| `Eigen`        | 고유치/고유벡터         | `time_values` → Mode ID |

```python
class HWASCIIExporter(BaseExporter):
    # 지원되지 않는 결과 타입은 WHTExportWarning을 발생시키고 건너뜀
    SUPPORTED_BLOCKS = {"Displacement", "Stress", "Strain", "Eigen"}

    def export(self, data: WHTResultData, output_path: str) -> None:
        ...
```

---

## 3. Error Handling Strategy ← NEW

```
솔버 출력
   │
   ▼
[Adapter.convert()]
   │
   ▼
WHTResultData
   │
   ▼
[Adapter.validate()]  ──── WHTValidationError → 즉시 중단
   │
   ▼
[Exporter.export()]   ──── WHTExportWarning  → 경고 후 계속
   │
   ▼
출력 파일
```

| 예외 클래스         | 발생 조건                              | 동작          |
|---------------------|----------------------------------------|---------------|
| `WHTValidationError`| 형상 불일치, 단위 미정의, 빈 배열 등   | 즉시 중단     |
| `WHTExportWarning`  | 지원되지 않는 필드 스킵 등             | 경고 후 계속  |

---

## 4. Implementation Plan

| 단계 | 파일                  | 산출물                                          |
|------|-----------------------|-------------------------------------------------|
| 1    | `wht_metadata.py`     | `WHTMetadata` 데이터클래스                      |
| 2    | `wht_result_data.py`  | `WHTResultData` + `WHTValidationError`          |
| 3    | `wht_adapters.py`     | `BaseAdapter`, `JaxSSOAdapter`, `JaxFEMAdapter` |
| 4    | `wht_exporters.py`    | `BaseExporter`, VTU/PVD, VTKHDF, HWASCII 익스포터 |
| 5    | `exam1_nf.py` 리팩터링 | 신규 컴포넌트 기반으로 파이프라인 재작성        |
| 6    | `tests/`              | 단위 테스트 + 교차 호환성 검증                  |

### 4.1 Test Strategy ← NEW

```
tests/
├── test_adapters.py       # JaxSSO/jax-fem → WHTResultData 변환 검증
├── test_exporters.py      # 각 포맷 출력 파일의 구조/값 검증
├── test_roundtrip.py      # 솔버 → 변환 → 파일 로드 → 값 일치 확인
└── fixtures/              # 소규모 벤치마크 메시 (2D/3D 혼합 요소 포함)
```

---

## 5. Key Benefits

| 항목 | 설명 | 상태 |
|------|------|------|
| **Zero Redundancy** | 메시 데이터는 VTKHDF에 1회만 저장 | 원안 유지 |
| **Single File** | 수백 개의 VTU 파일 대신 단일 `.hdf` 파일로 관리 | 원안 유지 |
| **Library Agnostic** | 솔버 교체 시 어댑터만 교체, 익스포터 코드 변경 불필요 | 원안 유지 |
| **Mixed Mesh Support** | Hex/Tet/Wedge 혼합 요소 메시를 정확히 표현 | ← NEW |
| **Extensible** | `BaseAdapter` / `BaseExporter` ABC로 신규 솔버·포맷을 독립적으로 추가 가능 | ← NEW |
| **Early Failure** | 단계별 검증으로 후처리 단계의 조용한 오류(silent bug) 방지 | ← NEW |

---

## 6. Change Summary (v0.1 → v0.2)

| 항목 | v0.1 (원안) | v0.2 (개정) |
|------|-------------|-------------|
| 메타데이터 | 없음 | `WHTMetadata` 분리 |
| 요소 연결 방식 | `elements: (M, V)` | `connectivity + offsets` (혼합 메시 지원) |
| Named Sets | 없음 | `node_sets`, `element_sets` 추가 |
| 전역 결과 | 없음 | `field_data` 추가 |
| 어댑터 구조 | 독립 함수 | `BaseAdapter` ABC + 구현체 클래스 |
| 익스포터 구조 | 독립 함수 | `BaseExporter` ABC + 구현체 클래스 |
| HWASCII 지원 블록 | 미명시 | `Displacement`, `Stress`, `Strain`, `Eigen` 명시 |
| VTKHDF 옵션 | 없음 | `compression`, `chunk_timesteps` 추가 |
| 오류 처리 | 없음 | `WHTValidationError` / `WHTExportWarning` 정의 |
| 테스트 전략 | "Validation" 1줄 | 4개 테스트 파일 + fixtures 구조 정의 |
| 구현 단계 | 4단계 | 6단계 (파일·산출물 명확화) |

---

*Please review this proposal and provide your feedback or approval.*
