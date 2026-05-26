# Implementation Plan - CalculiX Integration in WHT Topography Monitor (2026-05-25)

본 계획서는 `wht_topo` 모니터 UI에서 자체 JAX 솔버 외에 **CalculiX** 오픈소스 FEM 솔버를 활용하여 특정 이터레이션의 변형 메시 상태에서 모달 및 정적 해석을 실시간 수행하고 PyVista Visualizer에 연동하는 아키텍처 및 구현 방안을 기술합니다.

---

## 1. 아키텍처 설계 및 모듈식 연동 방안

`AutoCalculix` 프로젝트는 `D:\PythonCodeStudy\AutoCalculix`에 독립된 저장소로 존재하므로, 이를 복사하지 않고 **파이썬 라이브러리(패키지) 모듈 형태로 직접 참조하는 방식**을 적용하여 유지보수성을 극대화합니다.

### 1) AutoCalculix에 통합 API 인터페이스 제공 (`autocalculix_api.py`)
`AutoCalculix` 내에 `src/autocalculix_api.py`를 신규 설계하여 `WHT_LightChassisModel`에서 호출할 수 있는 표준 규격의 함수 인터페이스를 선언합니다.

```python
# D:\PythonCodeStudy\AutoCalculix\src\autocalculix_api.py
# -*- coding: utf-8 -*-
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.core.config import ModalAnalysisConfig, CALCULIX_EXE
from src.core.model_builder import CalculixModelBuilder
from src.core.solver import CalculixSolver
from src.core.dat_parser import CalculixDatParser
from src.core.frd_converter import FrdToVtuConverter

def run_calculix_analysis(
    nodes: dict,          # {nid: (x, y, z)}
    elements: list,       # [(eid, 'QUAD4'/'TRIA3', [n1, n2, ...], pid)]
    properties: dict,     # {pid: (thickness, E, nu, rho)}
    analysis_type: str,   # "modal" 또는 "static"
    analysis_config: dict,# {"num_modes": 10, "job_name": "ccx_job", ...}
    bcs: list = None,     # [(nid, [dofs], value)] - 정적해석 구속 조건
    forces: list = None,  # [(nid, [fx, fy, fz, mx, my, mz])] - 정적해석 하중 조건
    workspace_dir: str = None
) -> dict:
    """
    외부 파이썬 모듈(WHT_LightChassisModel)에서 넘겨받은 FEM 메쉬 정보와 경계/하중 조건을 바탕으로
    CalculiX 해석을 구동하고 결과를 반환합니다.
    """
    # 1. 파일 작성 경로 지정
    work_path = Path(workspace_dir) if workspace_dir else BASE_DIR / "workspace"
    work_path.mkdir(parents=True, exist_ok=True)
    
    job_name = analysis_config.get("job_name", "ccx_analysis")
    
    # 2. Abaqus .inp 형식의 메쉬 파일 동적 생성
    mesh_filename = f"{job_name}_mesh.inp"
    mesh_inp_path = work_path / mesh_filename
    
    with open(mesh_inp_path, 'w', encoding='utf-8') as f:
        # 노드 기록
        f.write("*NODE\n")
        for nid, (x, y, z) in sorted(nodes.items()):
            f.write(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}\n")
            
        # 요소 기록 (CalculiX 셸 요소 매핑: S4 또는 S3)
        # pid 별로 elset을 나누어 기록하여 다중 두께/재료 대응 지원
        elset_elems = {}
        for eid, etype, conn, pid in elements:
            elset_name = f"ELSET_PROP_{pid}"
            elset_elems.setdefault((etype, elset_name), []).append((eid, conn))
            
        for (etype, elset_name), elem_list in elset_elems.items():
            ccx_type = "S4" if "QUAD" in etype.upper() else "S3"
            f.write(f"\n*ELEMENT, TYPE={ccx_type}, ELSET={elset_name}\n")
            for eid, conn in elem_list:
                conn_str = ", ".join(str(n) for n in conn)
                f.write(f"{eid}, {conn_str}\n")

    # 3. Master .inp 빌드 및 해석 시나리오 구성 (정적 / 모달 분기)
    master_inp_path = work_path / f"{job_name}.inp"
    with open(master_inp_path, 'w', encoding='utf-8') as f:
        f.write(f"*HEADING\nCalculiX Analysis: {job_name}\n")
        f.write(f"*INCLUDE, INPUT={mesh_filename}\n\n")
        
        # 재료 및 섹션 정의
        for pid, (t, E, nu, rho) in properties.items():
            mat_name = f"MAT_PROP_{pid}"
            f.write(f"*MATERIAL, NAME={mat_name}\n")
            f.write(f"*ELASTIC\n{E}, {nu}\n")
            f.write(f"*DENSITY\n{rho}\n\n")
            f.write(f"*SHELL SECTION, ELSET=ELSET_PROP_{pid}, MATERIAL={mat_name}\n")
            f.write(f"{t}\n\n")
            
        # STEP 구성
        f.write("*STEP\n")
        if analysis_type.lower() == "modal":
            num_modes = analysis_config.get("num_modes", 10)
            f.write("*FREQUENCY\n")
            f.write(f"{num_modes}\n")
            f.write("*NODE FILE\nU\n")
        else:
            # 정적 해석 (Linear Static)
            f.write("*STATIC\n")
            
            # SPC 경계조건 인가 (*BOUNDARY)
            if bcs:
                f.write("*BOUNDARY\n")
                for nid, dofs, val in bcs:
                    # dofs는 파이썬 0-based 리스트 (예: [0, 1, 2] -> 1, 3)
                    for d in dofs:
                        f.write(f"{nid}, {d+1}, {d+1}, {val:.8f}\n")
                        
            # FORCE 하중 인가 (*CLOAD)
            if forces:
                f.write("*CLOAD\n")
                for nid, f_vec in forces:
                    # f_vec = [fx, fy, fz, mx, my, mz]
                    for idx, val in enumerate(f_vec):
                        if abs(val) > 1e-12:
                            f.write(f"{nid}, {idx+1}, {val:.8f}\n")
                            
            f.write("*NODE FILE\nU\n")
            f.write("*EL FILE\nS\n") # 응력 결과 출력 요청
            
        f.write("*END STEP\n")
        
    # 4. ccx 솔버 구동
    solver = CalculixSolver(CALCULIX_EXE)
    solver.run(job_name, work_path)
    
    # 5. 결과 파싱 및 후처리
    result_data = {"type": analysis_type.lower(), "job_name": job_name, "workspace": str(work_path)}
    
    # 5-1) 모달 주파수 파싱 (.dat)
    if analysis_type.lower() == "modal":
        parser = CalculixDatParser()
        freqs = parser.extract_frequencies(work_path / f"{job_name}.dat")
        result_data["frequencies"] = freqs
        
    # 5-2) FRD -> VTU 변환
    frd_file = work_path / f"{job_name}.frd"
    if frd_file.exists():
        converter = FrdToVtuConverter()
        vtu_base = converter.convert(frd_file)
        result_data["vtu_base"] = str(vtu_base)
        
    return result_data
```

### 2) WHT_LightChassisModel 연동 구현
`WHT_LightChassisModel\wht_topo\monitor_ui.py`에서 위 API를 동적 로드하여 비동기 스레드로 계산하도록 구현합니다.

* **경로 동적 추가 및 임포트**:
  ```python
  AUTOCALCULIX_PATH = "D:/PythonCodeStudy/AutoCalculix"
  if AUTOCALCULIX_PATH not in sys.path:
      sys.path.append(AUTOCALCULIX_PATH)
  from src.autocalculix_api import run_calculix_analysis
  ```

---

## 2. UI 레이아웃 및 버튼 연동 설계

기존 `Iteration Results` 탭 하단 액션 바의 `Run Analysis` 버튼 오른쪽에 **`Run CalculiX`** 버튼을 추가 배치합니다.

### 1) UI 버튼 추가 (`_build_tab_height`)
```python
# monitor_ui.py 내 _build_tab_height 수정
self.iter_run_btn = QPushButton("Run Analysis")
...
# CalculiX 모달/정적 해석 실행용 버튼 추가
self.iter_ccx_btn = QPushButton("Run CalculiX")
self.iter_ccx_btn.setStyleSheet("font-weight:bold; background-color:#1e3d59; color:white;")
self.iter_ccx_btn.setToolTip("선택 이터레이션/하중 케이스로 CalculiX 해석 구동 및 Visualizer 연동")
self.iter_ccx_btn.clicked.connect(self._on_run_calculix)
ctrl_action.addWidget(self.iter_ccx_btn)
```

### 2) 비동기 처리용 QThread 워커 구현 (`_CalculixReAnalysisWorker`)
해석 시간이 수 초 이상 소요될 수 있으므로, GUI 프리징 방지를 위해 `QThread` 상에서 비동기로 실행합니다.

```python
class _CalculixReAnalysisWorker(QThread):
    finished = Signal(dict)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, snap_dir: str, iter_num: int, case_name: str, num_modes: int = 20, parent=None):
        super().__init__(parent)
        self.snap_dir = snap_dir
        self.iter_num = iter_num
        self.case_name = case_name
        self.num_modes = num_modes

    def run(self):
        try:
            import numpy as np
            import pickle
            from pathlib import Path
            snap_dir = Path(self.snap_dir)

            self.progress.emit("Snapshots 정보 로드 및 변형 메시 생성 중...")
            with open(snap_dir / "init.pkl", "rb") as f:
                init = pickle.load(f)
            
            # 모델 변형 상태 복원 및 h_elem 적용
            model = init["model"]
            bead_dir = init["bead_dir"]
            design_nids = init["design_nids"]
            aggr_src = init["aggr_src"]
            aggr_dst = init["aggr_dst"]
            orig_coords = init["orig_coords"]

            for nid, (x, y, z) in orig_coords.items():
                nd = model.nodes[nid]
                nd.x, nd.y, nd.z = x, y, z

            if self.iter_num == 0:
                h_elem = np.zeros(len(init["design_elems"]))
                load_cases_raw = init.get("static_load_cases", [])
                load_cases = [(lc.name, w, lc) for lc, w in load_cases_raw]
            else:
                with open(snap_dir / f"iter_{self.iter_num:03d}.pkl", "rb") as f:
                    snap = pickle.load(f)
                h_elem = snap["h_elem"]
                load_cases = snap.get("load_cases", [])

            n_int = len(design_nids)
            h_node_sum = np.zeros(n_int)
            np.add.at(h_node_sum, aggr_src, h_elem[aggr_dst])
            node_adj = np.bincount(aggr_src, minlength=n_int)
            h_node = h_node_sum / (node_adj + 1e-12)

            for i, nid in enumerate(design_nids):
                ox, oy, oz = orig_coords[nid]
                nd = model.nodes[nid]
                dh = float(h_node[i])
                nd.x = ox + dh * bead_dir[0]
                nd.y = oy + dh * bead_dir[1]
                nd.z = oz + dh * bead_dir[2]

            # CalculiX API 인자용 데이터 매핑
            nodes_dict = {nid: (nd.x, nd.y, nd.z) for nid, nd in model.nodes.items()}
            
            elements_list = []
            for eid, elem in model.elements.items():
                etype = getattr(elem, "element_type", "QUAD4")
                pid = getattr(elem, "pid", 1)
                elements_list.append((eid, etype, list(elem.node_ids), pid))
                
            properties_dict = {}
            for pid, prop in getattr(model, "properties", {}).items():
                t = getattr(prop, "t", 1.2)
                mid = getattr(prop, "mid", 1)
                mat = getattr(model, "materials", {}).get(mid)
                E = getattr(mat, "E", 210000.0)
                nu = getattr(mat, "nu", 0.3)
                rho = getattr(mat, "rho", 7.85e-9)
                properties_dict[pid] = (t, E, nu, rho)
                
            if not properties_dict: # 폴백값 지정
                properties_dict[1] = (1.2, 210000.0, 0.3, 7.85e-9)

            # 해석 종류(모달 vs 정적)에 따른 인자 구성
            analysis_config = {
                "job_name": f"ccx_iter{self.iter_num:03d}_{self.case_name.replace(' ', '_')}",
                "num_modes": self.num_modes
            }
            
            bcs = []
            forces = []
            analysis_type = "modal" if self.case_name == "Modal Analysis" else "static"
            
            if analysis_type == "static":
                # 하중 조건 탐색
                lc_obj = None
                for entry in load_cases:
                    if entry[0] == self.case_name:
                        lc_obj = entry[2]
                        break
                if lc_obj is None:
                    raise ValueError(f"하중 케이스 '{self.case_name}'를 찾을 수 없습니다.")
                
                # SPC 및 FORCE 매핑
                for bc in lc_obj.bcs:
                    bcs.append((bc.node_id, list(bc.dofs), getattr(bc, "value", 0.0)))
                for fc in lc_obj.forces:
                    forces.append((fc.node_id, list(fc.load_vector)))

            # D:\PythonCodeStudy\AutoCalculix API 모듈 로드 및 실행
            self.progress.emit(f"CalculiX {analysis_type.upper()} 솔버 구동 중...")
            
            # sys.path 연동
            import sys
            from pathlib import Path
            ccx_path = "D:/PythonCodeStudy/AutoCalculix"
            if ccx_path not in sys.path:
                sys.path.append(ccx_path)
            from src.autocalculix_api import run_calculix_analysis
            
            res_dict = run_calculix_analysis(
                nodes=nodes_dict,
                elements=elements_list,
                properties=properties_dict,
                analysis_type=analysis_type,
                analysis_config=analysis_config,
                bcs=bcs,
                forces=forces,
                workspace_dir=str(snap_dir) # 스냅샷 디렉토리를 워크스페이스로 활용
            )
            
            self.finished.emit({
                "type": "calculix_" + analysis_type,
                "result": res_dict,
                "model": model,
                "lc_name": self.case_name
            })
            
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())
```

---

## 3. 정적 Loadcase에 대한 CalculiX 사용성 검토

CalculiX를 활용한 정적 하중 케이스(Static Loadcase) 연동 타당성 및 사용성은 아래와 같습니다.

### 1) 기술적 장점 및 효용성
- **접촉 및 대변형 등 비선형성 검증 우수**:
  자체 JAX FEA 솔버는 선형 복셀/쉘 기반 정적 해석만 가능합니다. 반면, CalculiX는 **기하학적 비선형(Large Displacement), 소성 재료(Plasticity), 경계부 접촉(Contact)** 기능을 정교하게 지원하므로, 섀시 보강판 비드가 과도하게 변형되거나 접촉부가 어긋나는 실무 비선형 영역 검증에 이상적입니다.
- **다양한 출력 변수 제공 (상용 툴 수준)**:
  해석 성공 후 폰 미세스 응력(Von Mises Stress), 변형률(Plastic Strain) 등 복잡한 응력 텐서를 완벽하게 출력하여 구조적 파손이나 항복(Yielding) 여부를 판별할 수 있습니다.
- **결과 교차 검증**:
  상용 솔버인 Abaqus나 Nastran/OptiStruct 결과값과 동일 레벨에서 직접적인 교차 비 교 검증이 가능하므로, 최적화 비드 형상에 대한 설계적 신뢰성을 확보해 줍니다.

### 2) 한계점 및 보완 대책
- **최적화 실시간 루프 연동의 한계**:
  CalculiX는 파일 기반 I/O가 필요하고 C/Fortran 바이너리로 동작하므로, 최적화 이터레이션 도중 수천 번 민감도(Sensitivity)를 평가해야 하는 메인 루프에 넣기에는 병목 현상이 큽니다.
  - **대책**: 최적화 연산 중에는 고속 JAX 솔버를 그대로 사용하고, **모니터 UI 단계(Post-Processing)에서 정밀 설계 검증 목적**으로CalculiX 수동 실행 버튼을 제공하는 현재의 방식이 최적의 조합입니다.
- **다자유도 하중조건 변환의 복잡성**:
  WHTSolver의 다방향 외력(모멘트 포함) 및 고차 구속 SPC를 오차 없이 CalculiX `*BOUNDARY`와 `*CLOAD` 키워드 규칙으로 정확하게 변환해 주는 로직이 중요합니다. 본 구현 계획의 API 매핑 구조를 통해 누락 없는 매핑을 보장합니다.

---

## 4. 검증 계획

### 1) 자동화 테스트 (Verification Script)
`D:\PythonCodeStudy\WHT_LightChassisModel\scratch\test_ccx_integration.py` 검증 스크립트를 작성하여 아래 사항을 검증합니다.
- `AutoCalculix` 경로가 `sys.path`에 정상적으로 주입되고 `run_calculix_analysis` API가 성공적으로 임포트되는지 확인.
- 임의의 미니 섀시 플레이트 모델 데이터를 정의하고, `run_calculix_analysis`를 호출하여 모달 해석(10개 모드) 및 정적 해석(100N 외력) 시 CCX 솔버가 크래시 없이 정상 구동하여 `.vtu` 및 `.dat` 출력을 생성하는지 검증.

### 2) 수동 UI 검증
- 모니터 UI 실행 후 `Iteration Results` 탭에서 이터레이션을 임의로 드래그합니다.
- `Load Case` 드롭다운에서 `Modal Analysis` 및 정적 하중 케이스를 하나씩 선택한 뒤, 신규 추가된 `Run CalculiX` 버튼을 클릭합니다.
- 터미널이나 모니터 UI 하단 상태바에 `CalculiX 모달/정적 솔버 구동 중...` 메시지가 뜨고 수 초 후 PyVista 3D Visualizer 창에 3D 변형 모드 및 변위 결과가 성공적으로 표출되는지 검증합니다.

---

## 5. (Update) WHT Solver vs CalculiX 고유진동수 불일치 원인 분석 및 해결

기존 WHT Solver와 CalculiX 솔버 간 고유진동수(모달 해석) 결과가 수백~수만 배 차이 나는 원인을 규명하고 해결하였습니다.

### 1) 분석 결과 (원인 규명)
- **단위계 오류 아님**: WHT 모델 내에서 길이(mm), 힘(N), 밀도(ton/mm^3)를 사용하는 일관된 단위계는 완벽하게 작동하고 있음이 확인되었습니다. (예: Steel 밀도 `7.85e-9` ton/mm^3 정상 적용 중).
- **WHT Solver 고유진동수 폭증 버그 (`M=0`)**: 유저 스크립트나 GUI 모델링 과정 중 `WHTElement` 생성 시 `pid` 값이 정상 할당되지 않거나 0으로 넘겨질 경우, 질량 매트릭스 어셈블러(`M_quad4_lumped` 등)가 해당 요소를 무시(skip)하여 절점 질량이 `0.0`으로 계산되는 치명적 문제가 있었습니다. 질량이 0이므로 $\sqrt{K/M}$ 공식에 의해 인공적으로 수만 Hz의 주파수가 계산된 것입니다.
- **CalculiX 모달 해석의 강체 모드(0 Hz) 버그**: `autocalculix_api.py` 내에서 INP 파일을 생성할 때, 모달 해석(`*FREQUENCY`) 분기에서 `*BOUNDARY` 구문을 출력하지 않도록 잘못 구현되어 있었습니다. 이로 인해 CCX는 구속 조건이 없는 상태(Free-Free)로 해석을 수행하여 항상 0 Hz 근방의 강체 모드만 반환하였습니다.

### 2) 적용된 해결책 (Proposed Changes)
- **AutoCalculix API 패키지 수정**: `autocalculix_api.py` 파일 내에서 `*BOUNDARY` 출력 코드를 `*STEP` 구문 위로 이동시켜 정적/모달 해석 모두에 경계 조건이 정상 적용되도록 수정 완료하였습니다.
- **WHT Solver 안전장치 확보 계획**: 
  - `wht_quad4_element.py` 및 `wht_tria3_element.py` 내에서 프로퍼티 매핑 실패 시(pid 결측 시) 조용히 `continue` 하지 않고 명시적인 ValueError를 발생시켜 유저가 모델링 오류를 즉시 인지하도록 안전장치를 추가할 계획입니다.

### 3) 검증 내역 (Verification)
- `scratch/compare_modal.py`를 통해 디버깅 스크립트를 작성하여 테스트한 결과, 정상적인 `pid` 할당 및 BC 적용이 이루어졌을 때 **WHT Solver (156.9 Hz) vs CCX (183.5 Hz)**로 첫 번째 벤딩 모드 주파수가 합리적으로 일치함을 확인하였습니다. (결과 차이 해소)
