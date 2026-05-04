# Generative AI-Driven Chassis Bead Design Tool: Detailed Implementation Plan

본 계획서는 `jax-fem` 기반 위상 최적화 엔진 구축부터 `jraph` 기반 GNN 생성형 AI 모델 개발까지, 향후 AI 에이전트(예: Gemini Flash)가 오차 없이 코드를 구현할 수 있도록 작성된 **상세 기술 명세 및 개발 지침서**입니다.

---

## 1. Directory Structure & Module Design (디렉토리 및 모듈 구조)

기존 `test_jaxSSO` 안에 두지 않고, 프로젝트 루트(`WHT_LightChassisModel/`) 바로 아래에 2개의 핵심 패키지를 신설합니다. 각 모듈은 철저하게 분리(Decoupling)되어 독립적인 테스트가 가능해야 합니다.

```text
WHT_LightChassisModel/
├── wht_topo/               # 위상 최적화 및 하중/제약조건 엔진 (독립 실행 가능)
│   ├── __init__.py
│   ├── solver.py           # JaxTopoSolver (SIMP + JAX-FEM Core)
│   ├── loads.py            # StochasticLoadManager (복합 하중 생성)
│   ├── constraints.py      # ModeTracker, DynamicConstraint (MAC 기반)
│   └── run_topo.py         # [신규] 단독 최적화 해석용 CLI/UI 엔트리포인트
│
├── wht_ai/                 # 대규모 데이터 생성 및 AI 학습/추론 엔진
│   ├── __init__.py
│   ├── data_gen.py         # DatasetOrchestrator (대량 병렬 생성)
│   ├── encoder.py          # DataEncoder (FEM Mesh -> jraph.GraphsTuple)
│   ├── gnn_model.py        # BeadGraphGenerativeModel (GAT 아키텍처)
│   └── inverse_design.py   # InverseDesigner (최종 생성 추론 파이프라인)
```

---

## 2. Core Class Interfaces (핵심 클래스 인터페이스 명세)

AI가 코드를 구현할 때 기준이 될 Input/Output 명세입니다.

### 2.1 `wht_topo.solver.JaxTopoSolver`
*   **역할**: JAX 기반의 고속 위상 최적화 수행.
*   **Input**: 
    *   `model`: `WHTMeshModel` 인스턴스
    *   `load_cases`: `loads.py`에서 생성된 다중 하중 텐서 `Dict[case_id, np.ndarray]`
    *   `constraints`: `constraints.py`에서 생성된 주파수 하한선 객체
*   **Output**: 노드별 최종 밀도 배열 `np.ndarray` (0.0 ~ 1.0)
*   **Algorithm**: SIMP (Solid Isotropic Material with Penalization) 방식 사용. JAX의 `@jax.jit`와 `value_and_grad`를 활용하여 Compliance 최소화 루프 구현.

### 2.2 `wht_topo.loads.StochasticLoadManager`
*   **역할**: 학습 일반화를 위한 복합 랜덤 하중 생성.
*   **Logic**: 
    *   기본 모드(Bending, Twisting, One-corner Lift)의 가중치를 무작위 할당 (예: 0.6*B + 0.3*T + 0.1*L).
    *   무작위 위치(노드)에 Point Load 추가 발생 기능.

### 2.3 `wht_topo.constraints.ModeTracker`
*   **역할**: 고유치 해석 결과에서 특정 모드(Bending/Twisting)를 식별.
*   **Algorithm**: MAC (Modal Assurance Criterion) 수식을 JAX로 구현하여, 기준 모드 형상(Reference Eigenvector)과 현재 이터레이션의 모드 형상을 내적 비교하여 모드 트래킹.

### 2.4 `wht_ai.encoder.DataEncoder`
*   **역할**: FEM 데이터를 `jraph` 학습용 그래프 구조로 인코딩.
*   **Node Features**: `[x, y, z, thickness, Fx, Fy, Fz]`
*   **Edge Features**: `[length, dx, dy, dz]`
*   **Globals**: `[target_freq_bending, target_freq_twisting]`
*   **Target (Node)**: `[optimized_density]`

---

## 3. Step-by-Step Execution Plan (단계별 구현 지침)

AI 에이전트는 다음 순서대로 각 단계를 구현하고 검증(Test)해야 합니다. 한 단계가 완벽히 동작하기 전에는 다음 단계로 넘어가지 않습니다.

### Phase 1: JAX-FEM Topo Solver Foundation (위상 최적화 기반 구축 및 단독 툴화)
*   **목표**: AI 데이터 생성용 엔진을 넘어, **그 자체로도 실무 적용이 가능한 고성능 스탠드얼론(Standalone) 위상 최적화 프로그램**을 개발하라.
1.  **`loads.py` 작성**: `StochasticLoadManager`를 구현하고, 생성된 하중 벡터를 PyVista로 시각화하여 물리적 정합성을 검증하라.
2.  **`solver.py` 작성**: 단일 Bending 하중에 대해 SIMP 기반의 Compliance 최소화 JAX 루프를 작성하라. 결과 밀도(Density)가 1과 0으로 뚜렷이 구분되는지 확인하라.
3.  **`constraints.py` 통합**: 고유치 해석 로직을 추가하고, MAC 기반 트래킹을 도입하여 "Twisting Mode > 40Hz" 페널티가 정상 작동하는지 테스트하라.
4.  **`run_topo.py` 작성 (단독 실행기)**: 사용자가 터미널에서 하중과 목표 진동수를 입력하면, 실시간으로 최적화를 수행하고 그 결과를 PyVista 시각화 창과 LS-DYNA(`.k`) 파일로 즉각 내보내는 독립 파이프라인을 완성하라.

### Phase 2: Mass Data Production (대규모 데이터 생성)
1.  **`data_gen.py` 작성**: `ChassisParametricMesh`의 폭(W)과 길이(L)를 루프를 돌며 무작위로 변경하는 스크립트를 작성하라.
2.  **Orchestration**: 각 형상에 대해 랜덤 하중을 가하고 `JaxTopoSolver`를 실행하라.
3.  **Storage**: 결과 밀도 맵과 메타데이터를 `HDF5` 또는 `NumPy` 직렬화 포맷으로 `/dataset` 폴더에 일괄 저장하는 로직을 구현하라. (목표: 1000개 샘플 생성 시나리오 테스트)

### Phase 3: GNN AI Model Development (생성형 AI 학습 모델)
1.  **`encoder.py` 작성**: 저장된 `.h5` 데이터를 읽어 `jraph.GraphsTuple`로 변환하는 파이프라인을 구축하라.
2.  **`gnn_model.py` 작성**: `jraph`의 `GraphNetwork` 또는 `GAT` (Graph Attention Network) 아키텍처를 정의하라. 노드, 엣지, 글로벌 피처가 모두 업데이트되는 딥러닝 망을 구성하라.
3.  **Training Loop**: JAX/Optax를 이용해 MSE Loss (예측 밀도 vs 실제 최적화 밀도)를 최소화하는 학습 루프를 작성하고 과적합 여부를 모니터링하라.

### Phase 4: Generative Inverse Inference (역설계 추론 엔진)
1.  **`inverse_design.py` 작성**: 학습된 가중치 모델을 로드하는 기능을 구현하라.
2.  **User Flow 구현**: 새로운 섀시 치수와 하중 조건이 주어지면, 즉각적으로 GNN을 통과시켜(Inference) 노드별 밀도를 예측하는 함수를 작성하라.
3.  **Mapping & Export**: 예측된 연속 리브 밀도를 기반으로 메시의 Z축(높이)을 변경하고, 이를 `WHTVisualizer`로 띄워주며 LS-DYNA `.k` 파일로 저장하는 통합 스크립트를 완성하라.

---

## 4. Verification Checklists (AI 에이전트 검증 기준)

*   [ ] **Phase 1 Check**: `run_topo.py` 단독 실행 시 엔지니어가 실무에 바로 사용할 수 있을 만큼 시각화 및 `.k` 파일 Export가 완벽하게 동작하는가? JIT 캐싱이 잘 이루어지는가?
*   [ ] **Phase 2 Check**: 크기가 서로 다른 메시를 생성해도 노드 인덱싱 오류 없이 데이터가 추출되는가?
*   [ ] **Phase 3 Check**: GNN 학습 시 Loss 곡선이 안정적으로 수렴하는가? 크기가 다른 Graph 데이터를 배치(Batch) 처리할 때 패딩(Padding) 로직이 완벽한가?
*   [ ] **Phase 4 Check**: 생성된 리브 패턴이 물리적 위상 최적화 결과 대비 90% 이상의 형상 일치도(IOU)를 보이며, 생성에 걸리는 시간이 1초 미만인가?
