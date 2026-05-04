# -*- coding: utf-8 -*-
"""
exam3_autobead.py
=================
WHT FEM Framework — 토포그래피 자동 비드(Auto-Beads)를 포함한 통합 모드 해석 예제

개요:
---------
본 스크립트는 강성 최적화를 위해 "토포그래피 비드(Topography Beads)"가 적용된 
구조용 쉘(Shell) 및 솔리드(Solid) 트레이 모델에 대해 고정밀 모드 해석을 수행합니다.

주요 기능:
-------------
- 훅-폴드 플랜지(Hook-Fold Flange) 구조 생성 (Rise -> Inward -> Fall -> Inward).
- 자동 비드 생성: 바닥면 보강을 위한 무작위 대칭 비드 패턴 생성.
- 이중 솔버 지원: JaxSSO (표준 쉘 해석) 및 jax-fem (고정밀 솔리드 육면체 해석).
- 시각화 통합: WHT Visualizer를 통한 비드 높이 및 모드 형상 프리뷰.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 시각화 설정 (사용자 규칙 준수: 한글 폰트 및 9pt 폰트 크기)
plt.rcParams['font.size'] = 9

# JAX float64 설정 (고유치 해석의 수치적 안정성 확보를 위해 필수)
jax.config.update("jax_enable_x64", True) 

# 프로젝트 루트 경로 추가 (상위 디렉토리의 wht_modeler 등을 참조하기 위함)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_adapters import JaxFEMAdapter
from wht_visualizer.wht_visualizer import WHTVisualizer
from mesh_utils import generate_shell_tray, generate_solid_hexa_tray, apply_auto_beads


# ==============================================================================
# 0. 설정 및 데이터 스키마 (CONFIGURATION & SCHEMA)
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    쉘/솔리드 트레이 및 자동 비드 생성을 위한 통합 설정 클래스입니다.
    모든 기하학적 치수와 해석 파라미터를 체계적으로 관리합니다.

    Attributes:
        width (float): 트레이의 가로 너비 [mm].
        length (float): 트레이의 세로 길이 [mm].
        height (float): 트레이의 기본 높이 [mm].
        thickness (float): 쉘 해석 시 적용할 기본 두께 [mm].
        solve_mode (str): 해석 모드 ('shell'은 JaxSSO, 'solid'는 jax-fem 사용).
        mesh_size_xy (float): 평면 방향(XY) 메시 크기 [mm].
        mesh_size_z (float): 높이 방향(Z) 메시 크기 (플랜지부).
        draft_angle (float): 측벽의 구배 각도 [degree].
        wall_layers (int): 솔리드 육면체 해석 시 두께 방향 레이어 수.
        flanges (tuple): 각 변별 플랜지 생성 여부 (Bottom, Top, Left, Right 순).
        flange_segments (List[Tuple[float, float]]): 플랜지 형상을 정의하는 (dx, dz) 세그먼트 리스트.
        bead_mode (str): 비드 패턴 생성 모드 ('random' 또는 'grid').
        bead_margin (float): 트레이 외곽으로부터 비드가 생성되지 않는 여유 공간 [mm].
        bead_target_ratio (float): 비드가 차지할 면적 비율 (0.0 ~ 1.0).
        bead_min_depth (float): 비드의 최소 돌출/함몰 깊이 [mm].
        bead_max_depth (float): 비드의 최대 돌출/함몰 깊이 [mm].
        bead_min_size (float): 비드 패턴의 최소 크기 [mm].
        bead_max_size (float): 비드 패턴의 최대 크기 [mm].
        num_modes (int): 추출할 고유 진동 모드 수.
        solver_method (str): 모드 해석 솔버 메서드 ('auto', 'sparse' 등).
        exclude_rigid_body (Union[bool, str]): 강체 모드(Rigid Body Mode) 제외 여부.
        E (float): 영률(Young's Modulus) [MPa, N/mm^2].
        nu (float): 포아송 비(Poisson's Ratio).
        rho (float): 밀도(Density) [ton/mm^3].
    """
    # 1. 기하학적 치수 (Geometry Dimensions)
    width:  float = 1800.0
    length: float = 1200.0
    height: float = 35.0
    thickness: float = 0.6
    
    # 2. 메시 제어 (Mesh Control)
    solve_mode: str = 'shell' 
    mesh_size_xy: float = 20.0
    mesh_size_z:  float = 10.0
    draft_angle:  float = 25.0
    wall_layers:  int = 2
    
    # 3. 플랜지 구성 (Hook-Fold Sequence)
    flanges: tuple = (False, True, True, True) 
    flange_segments: List[Tuple[float, float]] = field(default_factory=list)
    
    # 4. 자동 비드 설정 (Auto-Bead/Topography Configuration)
    bead_mode: str = 'random'
    bead_margin: float = 50.0
    bead_target_ratio: float = 0.4
    bead_min_depth: float = 10.0
    bead_max_depth: float = 20.0
    bead_min_size: float = 150.0
    bead_max_size: float = 200.0
    
    bead_direction: float = 0.0 # 1.0: Up, -1.0: Down, 0.0: Both
    
    # 5. 솔버 및 해석 설정 (Solver & Analysis Settings)
    num_modes: int = 10
    solver_method: str = 'auto'
    exclude_rigid_body: Union[bool, str] = 'auto'
    
    # 6. 재료 물성 (Material: Steel)
    E:   float = 210000.0
    nu:  float = 0.3
    rho: float = 7.85e-9


# ==============================================================================
# 1. 해석 파이프라인 클래스 (ANALYSIS PIPELINE)
# ==============================================================================

class ModalAnalysisPipeline:
    """
    트레이 기하구조 생성부터 모드 해석, 시각화까지의 전 과정을 관리하는 클래스입니다.
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model: Optional[WHTMeshModel] = None
        self.node_db_orig: Dict = {}
        self.node_db_morphed: Dict = {}
        self.elem_db: Dict = {}
        self.results = None

    def run_geometry_stage(self):
        """기본 기하구조 생성 및 비드 적용 단계를 수행합니다."""
        if self.cfg.solve_mode == 'shell':
            print(f" -> 기본 쉘 트레이 메시 생성 중 (크기={self.cfg.mesh_size_xy})...")
            self.node_db_orig, self.elem_db = generate_shell_tray(
                width=self.cfg.width, length=self.cfg.length, height=self.cfg.height,
                mesh_size_xy=self.cfg.mesh_size_xy, mesh_size_z=self.cfg.mesh_size_z,
                draft_angle=self.cfg.draft_angle, flange_segments=self.cfg.flange_segments,
                flanges=self.cfg.flanges, mesh_type='quad4', origin='center'
            )
        else:
            print(f" -> 기본 솔리드 트레이 메시 생성 중 (두께={self.cfg.thickness})...")
            self.node_db_orig, self.elem_db = generate_solid_hexa_tray(
                width=self.cfg.width, length=self.cfg.length, height=self.cfg.height,
                mesh_size_xy=self.cfg.mesh_size_xy, mesh_size_z=self.cfg.mesh_size_z,
                draft_angle=self.cfg.draft_angle, wall_layers=self.cfg.wall_layers,
                flange_segments=self.cfg.flange_segments, flanges=self.cfg.flanges,
                origin='center', thickness=self.cfg.thickness
            )

        # 토포그래피 비드 적용 (KDTree 기반의 강건한 대칭 매핑 사용)
        print(" -> 대칭형 자동 비드(토포그래피 변형) 적용 중...")
        self.node_db_morphed = apply_auto_beads(
            node_db=self.node_db_orig, 
            width=self.cfg.width, length=self.cfg.length,
            margin=self.cfg.bead_margin,
            target_ratio=self.cfg.bead_target_ratio,
            min_depth=self.cfg.bead_min_depth,
            max_depth=self.cfg.bead_max_depth,
            min_size=self.cfg.bead_min_size,
            max_size=self.cfg.bead_max_size,
            origin='center',
            mode=self.cfg.bead_mode,
            bead_direction=self.cfg.bead_direction
        )

    def assemble_model(self):
        """WHT 모델을 구성하고 물성 및 경계 조건을 할당합니다."""
        is_solid = (self.cfg.solve_mode == 'solid')
        self.model = WHTMeshModel.from_node_elem_db(self.node_db_morphed, self.elem_db, is_solid=is_solid)
        self.model.name = f"{self.cfg.solve_mode.capitalize()}_Beaded_Tray"
        
        # 1. 재료 및 속성 정의
        self.model.add_material(1, self.cfg.E, self.cfg.nu, self.cfg.rho)
        if is_solid:
            self.model.add_property(1, "PSOLID", 0.0, 1)
        else:
            self.model.add_property(1, "PSHELL", self.cfg.thickness, 1)
        
        for elem in self.model.elements.values():
            elem.pid = 1
            
        # 2. 경계 조건 설정 (최상단 플랜지 고정)
        # 취약점 개선: 하드코딩된 max_z 대신 실제 노드 데이터에서 최상단 높이를 추출하여 선택
        all_z = [n.z for n in self.model.nodes.values()]
        actual_max_z = max(all_z)
        
        # [수정 5/5 — BC 허용오차를 mesh_size_z에 비례하도록 수정]
        # 이전 코드의 문제:
        #   `abs(node.z - actual_max_z) < 0.1`에서 0.1은 고정된 절대값이다.
        #   mesh_size_z가 0.1mm 이하로 설정되면 수치 오차 범위와 겹쳐
        #   최상단 노드가 경계조건에서 누락될 수 있다.
        #   반대로 mesh_size_z가 매우 크면 0.1mm가 지나치게 엄격하여
        #   실제로는 같은 면인 노드들이 오차 범위 밖으로 판정될 수 있다.
        #
        # 수정 방법:
        #   허용 오차를 cfg.mesh_size_z의 일정 비율(10%)로 설정한다.
        #   mesh_size_z가 변해도 "메시 한 칸보다 훨씬 작은 오차"라는 의미를 유지한다.
        #   최솟값 0.01을 두어 매우 정밀한 메시에서도 최소 오차를 보장한다.
        bc_tol = max(0.01, self.cfg.mesh_size_z * 0.1)
        fixed_count = 0
        for nid, node in self.model.nodes.items():
            if abs(node.z - actual_max_z) < bc_tol:
                self.model.apply_spc(nid, (0, 1, 2, 3, 4, 5))
                fixed_count += 1
        print(f"    [BC] 최상단 플랜지 위치({actual_max_z:.2f}mm) 노드 {fixed_count}개 구속 완료. (허용오차={bc_tol:.4f}mm)")

    def export_mesh(self, path: str):
        """
        메시를 LS-DYNA 포맷(.k)으로 내보냅니다.
        
        Args:
            path (str): 저장할 파일 경로 (예: 'bead_model.k')
        """
        if self.model is None:
            print("    [오류] 모델이 조립되지 않아 내보낼 수 없습니다.")
            return
            
        print(f" -> LS-DYNA 포맷으로 메시 내보내는 중: {path}...")
        try:
            self.model.export_to_solver('lsdyna', path)
        except Exception as e:
            print(f"    [오류] 메시 내보내기 실패: {e}")

    def solve(self):
        """솔버를 실행하여 결과를 도출합니다."""
        if self.cfg.solve_mode == 'shell':
            print(f" -> 모드 해석 수행 중 (JaxSSO Sparse)...")
            solver = WHTSolver(self.model)
            self.results = solver.solve_modal(
                num_modes=self.cfg.num_modes, 
                method=self.cfg.solver_method,
                exclude_rigid_body=self.cfg.exclude_rigid_body
            )
        else:
            self.results = self._solve_solid_jaxfem()

    def _solve_solid_jaxfem(self):
        """솔리드 전용 고성능 질량 행렬 연산 및 고유치 해석 루틴입니다."""
        print(f" -> 솔리드 육면체 해석 파이프라인 가동 (모드 수={self.cfg.num_modes})...")
        try:
            from jax_fem.generate_mesh import Mesh
            from jax_fem.problem import Problem
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import diags, csr_matrix
        except ImportError:
            raise ImportError("솔리드 해석을 위해 jax-fem 라이브러리가 필요합니다.")

        # jax-fem 메시 변환
        points = self.model.nodes_array()
        nid_to_idx = self.model.node_id_to_index()
        cells = np.array([[nid_to_idx[nid] for nid in self.model.elements[eid].node_ids] 
                         for eid in sorted(self.model.elements.keys()) if self.model.elements[eid].type == "HEXA8"])
        mesh = Mesh(points, cells)
        
        # [수정 2/5 — ModalSolidProblem.get_tensor_map 클로저 버그 수정]
        # 이전 코드의 문제:
        #   내부 클래스 ModalSolidProblem은 jax-fem의 Problem을 상속받으므로,
        #   get_tensor_map 내부의 `self`는 ModalSolidProblem 인스턴스를 가리킨다.
        #   그런데 `self.cfg`는 Problem 클래스에도, ModalSolidProblem에도 존재하지
        #   않으므로 AttributeError가 발생한다.
        #
        # 수정 방법:
        #   파이프라인의 cfg 객체를 클래스 정의 시점에 클로저(closure)로 캡처한다.
        #   즉, 지역변수 _E, _nu에 값을 미리 복사해두면, 런타임에 `self`를 통하지
        #   않고도 재료 상수에 접근할 수 있다.
        _E  = self.cfg.E   # 클로저 캡처: pipeline.cfg.E
        _nu = self.cfg.nu  # 클로저 캡처: pipeline.cfg.nu

        class ModalSolidProblem(Problem):
            def get_tensor_map(self):
                def constitutive(u_grad):
                    # _E, _nu는 외부 스코프에서 캡처된 값 — self.cfg 참조 불필요
                    mu    = _E / (2.0 * (1.0 + _nu))
                    lmbda = _E * _nu / ((1.0 + _nu) * (1.0 - 2.0 * _nu))
                    eps = 0.5 * (u_grad + u_grad.T)
                    return lmbda * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
                return constitutive

        # 경계 조건 필터 (actual_max_z 사용)
        actual_max_z = max(points[:, 2])
        def flange_filter(x): return jnp.isclose(x[2], actual_max_z, atol=0.1)
        dirichlet_bc = [[flange_filter]*3, [0, 1, 2], [lambda x: 0.0]*3]
        prob = ModalSolidProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc)
        
        # 행렬 어셈블리 및 효율적인 질량 계산
        print("    [Solid] 강성 및 집중 질량 행렬 조립 중 (Vectorized)...")
        prob.newton_update(prob.unflatten_fn_sol_list(np.zeros(prob.num_total_dofs_all_vars)))
        K = csr_matrix((np.array(prob.V), (prob.I, prob.J)), shape=(prob.num_total_dofs_all_vars, prob.num_total_dofs_all_vars))
        
        # 취약점 개선: PyVista 의존성을 제거하고 직접 요소 체적 기반 질량 계산
        # HEXA8 체적 계산 (5개의 사면체로 분할하여 합산)
        nodal_mass = np.zeros(len(points))
        for cell in cells:
            p = points[cell]
            # Hexa8을 5개의 사면체로 분할하는 표준 인덱스 (0-1-2-5, 0-2-3-7, 0-5-7-4, 2-5-7-6, 0-2-5-7)
            tetras = [
                (0, 1, 2, 5), (0, 2, 3, 7), (0, 5, 7, 4), (2, 5, 7, 6), (0, 2, 5, 7)
            ]
            v_total = 0.0
            for t in tetras:
                v_total += abs(np.linalg.det(p[list(t[1:])] - p[t[0]])) / 6.0
            
            nodal_mass[cell] += (v_total * self.cfg.rho) / 8.0 # 8개 노드에 균등 배분 (Lumped)
        
        # 실제로는 pyvista.compute_cell_sizes()가 가장 정확하므로 성능 이슈가 없다면 유지 가능하나,
        # 여기서는 의존성 제거를 위해 NumPy 기반으로 대체함.
        M = diags([np.repeat(nodal_mass, 3)], [0], shape=K.shape, dtype=K.dtype).tocsr()

        vals, vecs = eigsh(K, k=self.cfg.num_modes, M=M, which='LM', sigma=-0.1)
        res_vecs = vecs.reshape((len(points), 3, self.cfg.num_modes)).transpose(2, 0, 1)
        return (prob, {"eigvecs": res_vecs, "eigvals": vals})

    def visualize(self):
        """결과를 시각화합니다. 해석 결과가 없더라도 기하구조 및 비드 데이터는 표시합니다."""
        print("\n -> 시각화 데이터를 구성 중...")
        
        # 1. 메타데이터 설정
        solver_name = "JaxSSO" if self.cfg.solve_mode == 'shell' else "jax-fem"
        meta = WHTMetadata(
            solver_name=solver_name, 
            solver_version="2.1.0", 
            analysis_type="modal", 
            coordinate_system="cartesian",
            unit_length="mm", 
            unit_force="N"
        )

        # 2. 결과 데이터 생성 (해석 결과가 있는 경우와 없는 경우 구분)
        if self.results is not None:
            if self.cfg.solve_mode == 'shell':
                wht_data = self.results.to_wht_result_data(meta, self.model)
            else:
                prob, res_dict = self.results
                adapter = JaxFEMAdapter()
                wht_data = adapter.convert(prob, res_dict, "modal", meta)
        else:
            # 미리보기 모드용 더미 결과 데이터 생성
            wht_data = self.model.to_wht_result_data(meta)

        # 3. 비드 높이 데이터 주입
        wht_data = inject_bead_metadata(wht_data, self.model, self.node_db_orig, self.node_db_morphed)
        
        # 4. 뷰어 실행
        viz = WHTVisualizer(title=f"비드 트레이 시각화 [{self.cfg.solve_mode.upper()}]", show=True)
        viz.load_results(wht_data, color="black")
        viz.plotter.view_isometric()
        if hasattr(viz.plotter, 'app'): viz.plotter.app.exec_()


# ==============================================================================
# 2. 유틸리티 및 메인 루틴
# ==============================================================================

def inject_bead_metadata(wht_data: WHTResultData, model: WHTMeshModel, n_orig: Dict, n_new: Dict):
    """비드 높이 정보를 시각화 필드에 추가합니다."""
    dz_list = []
    for nid in model.sorted_node_ids():
        dz = n_new[nid][2] - n_orig[nid][2]
        dz_list.append(dz)
    num_t = len(wht_data.time_values) if wht_data.time_values is not None else 1
    dz_vector = np.zeros((num_t, len(dz_list), 3), dtype=np.float32)
    dz_vector[:, :, 2] = np.array(dz_list, dtype=np.float32).reshape(1, -1)
    wht_data.point_data["Bead_Height"] = dz_vector
    return wht_data

def main():
    parser = argparse.ArgumentParser(description="토포그래피 자동 비드 생성 및 통합 모드 해석 (개선판)")
    parser.add_argument("--solve", type=str, choices=['shell', 'solid'], default='shell')
    parser.add_argument("--mode", type=str, choices=['grid', 'random', 'rib', 'network'], default='network')
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export", type=str, help="메시를 저장할 파일 경로 (예: model.k)")
    parser.add_argument("--no-viz", action="store_true", help="시각화 창을 띄우지 않습니다.")
    parser.add_argument("--direction", type=str, choices=['up', 'down', 'both'], default='down', help="비드 생성 방향")
    args = parser.parse_args()

    dir_map = {'up': 1.0, 'down': -1.0, 'both': 0.0}
    
    hook_sequence = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
    cfg = PipelineConfig(
        solve_mode=args.solve, 
        bead_mode=args.mode, 
        flange_segments=hook_sequence,
        bead_direction=dir_map[args.direction]
    )

    print("\n" + "="*80)
    print(f" [시스템] {cfg.solve_mode.upper()} 파이프라인 가동 시작")
    print("="*80)

    try:
        pipeline = ModalAnalysisPipeline(cfg)
        pipeline.run_geometry_stage()
        pipeline.assemble_model()
        
        # 메시 내보내기 요청이 있는 경우 수행
        if args.export:
            pipeline.export_mesh(args.export)
        
        if args.preview:
            print(" -> [알림] 미리보기 모드입니다. 해석을 건너뛰고 형상을 시각화합니다.")
        else:
            pipeline.solve()
        
        if not args.no_viz:
            pipeline.visualize()
        else:
            print(" -> [알림] --no-viz 옵션에 의해 시각화를 건너뜁니다.")
            
    except Exception:
        print("\n" + "!"*80)
        print(" [오류] 해석 파이프라인 실행 중 치명적 결함이 발견되었습니다.")
        print("!"*80)
        traceback.print_exc()

if __name__ == "__main__":
    main()


