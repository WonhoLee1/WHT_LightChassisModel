import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
from wht_modeler.wht_selector import WHTSelector
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.load_cases import WHTLoadCase, LoadCaseLibrary

class StochasticLoadManager:
    """
    Generates stochastic complex load conditions for Topography Optimization and AI training.
    Supports mixing of basic mode shapes (Bending, Twisting, One-corner Lift) and random point loads.
    """
    def __init__(self, model: WHTMeshModel):
        """
        Args:
            model: WHTMeshModel instance.
        """
        self.model = model
        self.node_ids = sorted(model.nodes.keys())
        self.num_nodes = len(self.node_ids)
        
        # 샤시 치수 데이터 추출
        nodes_arr = model.nodes_array()
        self.x_min, self.x_max = nodes_arr[:, 0].min(), nodes_arr[:, 0].max()
        self.y_min, self.y_max = nodes_arr[:, 1].min(), nodes_arr[:, 1].max()
        self.z_min, self.z_max = nodes_arr[:, 2].min(), nodes_arr[:, 2].max()
        self.width = self.x_max - self.x_min
        self.length = self.y_max - self.y_min
        self.height = self.z_max - self.z_min
        
        self.coords = nodes_arr
        self.num_modes = 5  # 사용자 요청에 따라 5개 모드 고려

    def auto_setup_from_metadata(self, model: WHTMeshModel):
        """
        [지능형 매핑] 모델의 노드 세트 이름을 분석하여 자동으로 SPC 및 하중 적용
        """
        print(" -> [LoadManager] 메타데이터 분석 및 자동 매핑 시작...")
        
        # 1. 경계 조건 (SPC) 자동 매핑
        # 'mounting', 'fix', 'support', 'flange' 등의 키워드가 포함된 세트 탐색
        found_spc = False
        for sid, nset in model.node_sets.items():
            name = nset.name.lower()
            if any(key in name for key in ["mounting", "fix", "support", "spc"]):
                print(f"    - [SPC] 노드 세트 '{nset.name}' 감지 (ID: {sid}) -> 완전 고정 적용")
                model.apply_spc(nset.node_ids, (0, 1, 2, 3, 4, 5))
                found_spc = True
        
        if not found_spc:
            print("    - [경고] 적절한 SPC 노드 세트를 찾지 못했습니다. 기본 하부 고정을 수행합니다.")
            model.apply_spc(self.get_boundary_nodes(), (0, 1, 2, 3, 4, 5))

        # 2. 하중 영역 식별
        # 'floor', 'tray', 'load' 등의 키워드 탐색
        for sid, nset in model.node_sets.items():
            if any(key in nset.name.lower() for key in ["floor", "tray", "load_area"]):
                print(f"    - [LOAD] 하중 영역 '{nset.name}' 식별됨.")
                # 최적화 시 이 영역을 우선적으로 고려하도록 설정 가능

    def get_boundary_nodes(self, mesh_size_z: float = 10.0) -> List[int]:
        """
        최상단 플랜지(Flange/Rim) 노드 ID 리스트를 식별합니다. (exam3_autobead.py 로직 반영)
        모델의 실제 최대 높이와 메시 크기에 따른 가변 허용 오차를 사용하여 강건하게 선택합니다.
        """
        # 1. 실제 모델의 최대 높이 추출
        all_z = [node.z for node in self.model.nodes.values()]
        if not all_z: return []
        actual_max_z = max(all_z)
        
        # 2. 메시 크기에 비례하는 허용 오차 설정 (exam3_autobead.py 참고)
        # 메시 한 칸의 10% 정도를 오차 범위로 설정하여 수치적 오차에 대응
        bc_tol = max(0.01, mesh_size_z * 0.1)
        
        selector = WHTSelector(self.model)
        
        # 3. Z-Level 기반 최상단 영역 선택
        top_nids = selector.by_box(z=(actual_max_z - bc_tol, actual_max_z + bc_tol)).get_ids()
        
        # 4. 선택된 노드들이 플랜지의 일부분일 경우, 곡률 기반으로 전체 플랜지 면 확장
        if top_nids:
            # 35도 이내의 각도를 가진 인접면 노드들을 선택하여 플랜지 전체를 커버
            flange_nids = (WHTSelector(self.model)
                           .by_ids(top_nids)
                           .expand_by_face(angle_limit_deg=35.0, z_min=actual_max_z - 5.0)
                           .get_ids())
            return list(flange_nids)
        
        return list(top_nids)

    def get_load_nodes(self) -> List[int]:
        """
        하중이 주로 인가되는 노드(바닥면 센터 및 코너 등)를 식별하여 반환합니다.
        민감도 해석의 가이드라인으로 활용됩니다.
        """
        cx, cy = (self.x_min + self.x_max)/2, (self.y_min + self.y_max)/2
        selector = WHTSelector(self.model)
        
        # 중앙 근처 300mm 반경 노드 (바닥면 기준)
        center_nids = selector.by_box(
            x=(cx - 150, cx + 150),
            y=(cy - 150, cy + 150),
            z=(self.z_min - 0.1, self.z_min + 5.0)
        ).get_ids()
        
        return list(center_nids)

    def get_corner_nodes(self, corner_idx: int, radius: float = 80.0) -> List[int]:
        """
        바닥면의 특정 코너 근처 노드 리스트를 반환합니다.

        Parameters
        ----------
        corner_idx : int
            0=좌하(-x,-y), 1=우하(+x,-y), 2=좌상(-x,+y), 3=우상(+x,+y)
        radius : float
            코너 중심으로부터의 탐색 반경 [mm]

        Returns
        -------
        List[int]
            코너 근처의 노드 ID 리스트
        """
        cx = self.x_min if corner_idx in [0, 2] else self.x_max
        cy = self.y_min if corner_idx in [0, 1] else self.y_max
        selector = WHTSelector(self.model)
        corner_nids = selector.by_box(
            x=(cx - radius, cx + radius),
            y=(cy - radius, cy + radius),
            z=(self.z_min - 0.1, self.z_min + 5.0)
        ).get_ids()
        return list(corner_nids)

    def get_load_cases(
        self,
        mesh_size_z: float = 10.0,
        bending_load_z: float = -5000.0,
        twisting_load_z: float = -3000.0,
        lifting_load_z: float = 3000.0,
        weights: Dict[str, float] = None,
    ) -> List[Tuple[WHTLoadCase, float]]:
        """
        Bending, Twisting, Lifting 하중 케이스를 각각의 물리적으로 타당한
        경계 조건(BC)과 함께 WHTLoadCase 리스트로 반환합니다.

        각 하중 케이스는 wht_solver.WHTSolver.solve_static()으로 직접 해석 가능합니다.

        Parameters
        ----------
        mesh_size_z : float
            메시 크기(Z방향), 플랜지 노드 탐색 허용 오차 계산에 사용 [mm]
        bending_load_z : float
            굽힘 하중 (중앙 바닥면, 하향) [N]
        twisting_load_z : float
            비틀림 하중 (대각선 코너, 하향) [N]
        lifting_load_z : float
            리프팅 하중 (코너 리프팅, 상향) [N]
        weights : Dict[str, float]
            각 하중 케이스의 목적 함수 가중치 {"bending", "twisting", "lifting"}

        Returns
        -------
        List[Tuple[WHTLoadCase, float]]
            (WHTLoadCase, weight) 튜플 리스트
        """
        if weights is None:
            weights = {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}

        flange_nids = self.get_boundary_nodes(mesh_size_z=mesh_size_z)
        center_nids = self.get_load_nodes()
        corner0 = self.get_corner_nodes(0)  # 좌하 (-x, -y)
        corner1 = self.get_corner_nodes(1)  # 우하 (+x, -y)
        corner2 = self.get_corner_nodes(2)  # 좌상 (-x, +y)
        corner3 = self.get_corner_nodes(3)  # 우상 (+x, +y)

        load_cases = []

        # ── Case 1: Bending (굽힘) ──────────────────────────────────────────
        # BC: 플랜지 전체 고정 (모든 DOF)
        # LOAD: 바닥 중앙부에 하향 균등 분포 하중
        lc_bending = WHTLoadCase(name="bending")
        lc_bending.add_bc(flange_nids, dofs=(0, 1, 2, 3, 4, 5))
        if center_nids:
            lc_bending.add_force(
                center_nids, dofs=(2,),
                values=(bending_load_z,), distribute=True
            )
        print(f" -> [LoadCase] Bending: {len(flange_nids)}개 플랜지 고정, "
              f"{len(center_nids)}개 중앙 노드에 {bending_load_z:.0f}N 하중")
        load_cases.append((lc_bending, weights["bending"]))

        # ── Case 2: Twisting (비틀림) ───────────────────────────────────────
        # BC: 대각선 방향 두 코너(0번, 3번) 완전 고정
        # LOAD: 반대쪽 대각선 두 코너(1번, 2번)에 서로 반대 방향 Z 하중
        lc_twisting = WHTLoadCase(name="twisting")
        lc_twisting.add_bc(corner0 + corner3, dofs=(0, 1, 2, 3, 4, 5))
        if corner1:
            lc_twisting.add_force(corner1, dofs=(2,),
                                  values=(-twisting_load_z,), distribute=True)
        if corner2:
            lc_twisting.add_force(corner2, dofs=(2,),
                                  values=(twisting_load_z,), distribute=True)
        print(f" -> [LoadCase] Twisting: 대각 2코너 고정, "
              f"반대 2코너에 ±{abs(twisting_load_z):.0f}N 비틀림 하중")
        load_cases.append((lc_twisting, weights["twisting"]))

        # ── Case 3~6: Lifting (4개 코너 개별 리프팅) ──────────────────────────
        # 각 코너에 대해 3개 코너 고정 + 1개 코너 리프팅을 개별 케이스로 생성
        corners = [corner0, corner1, corner2, corner3]
        for i in range(4):
            lc_name = f"lifting_c{i}"
            lc_lift = WHTLoadCase(name=lc_name)
            
            # 리프팅 지점을 제외한 나머지 3개 코너 고정
            fixed_corners = []
            for j in range(4):
                if i == j: continue
                fixed_corners += corners[j]
            
            if fixed_corners:
                lc_lift.add_bc(fixed_corners, dofs=(0, 1, 2)) # Translation 고정
            
            # 대상 코너 리프팅 하중 인가
            target_corner = corners[i]
            if target_corner:
                lc_lift.add_force(target_corner, dofs=(2,), 
                                  values=(lifting_load_z,), distribute=True)
                
            print(f" -> [LoadCase] {lc_name}: {len(fixed_corners)}개 노드 고정, "
                  f"코너 {i}에 +{lifting_load_z:.0f}N 리프팅 하중")
            # 리프팅 가중치를 4개 케이스로 나누어 적용 (전체 비중 유지)
            load_cases.append((lc_lift, weights["lifting"] / 4.0))

        return load_cases


    def generate_random_load_case(self, 
                                  seed: int = None,
                                  base_modes_weight: float = 0.7,
                                  random_points_weight: float = 0.3,
                                  num_random_points: int = 5) -> jnp.ndarray:
        """
        복합 랜덤 하중 벡터 맵을 생성합니다. (단위: N)
        """
        if seed is not None:
            np.random.seed(seed)
            
        load_vector = np.zeros((self.num_nodes, 3))
        
        # 1. Base Modes 가중치 결정
        weights = np.random.dirichlet(np.ones(3)) 
        
        bending_load = self._generate_bending_pattern()
        twisting_load = self._generate_twisting_pattern()
        lifting_load = self._generate_lifting_pattern()
        
        base_load = (weights[0] * bending_load + 
                     weights[1] * twisting_load + 
                     weights[2] * lifting_load)
        
        load_vector += base_load * base_modes_weight
        
        # 2. Random Point Loads
        # 바닥면(z_min 부근) 노드들 중에서 랜덤 선택하여 국부 하중 가중
        floor_indices = np.where(self.coords[:, 2] < self.z_min + self.height * 0.2)[0]
        
        for _ in range(num_random_points):
            if len(floor_indices) > 0:
                idx = np.random.choice(floor_indices)
                f_rand = np.random.normal(0, 150, 3) 
                load_vector[idx] += f_rand * random_points_weight
            
        return jnp.array(load_vector)

    def _generate_bending_pattern(self) -> np.ndarray:
        loads = np.zeros((self.num_nodes, 3))
        cx, cy = (self.x_min + self.x_max)/2, (self.y_min + self.y_max)/2
        dist_sq = (self.coords[:, 0] - cx)**2 + (self.coords[:, 1] - cy)**2
        max_dist_sq = (self.width/2)**2 + (self.length/2)**2
        
        # Z축 하향 가우시안 분포 하중
        loads[:, 2] = -1000.0 * np.exp(-dist_sq / (0.3 * max_dist_sq))
        return loads

    def _generate_twisting_pattern(self) -> np.ndarray:
        loads = np.zeros((self.num_nodes, 3))
        cx, cy = (self.x_min + self.x_max)/2, (self.y_min + self.y_max)/2
        tx = (self.coords[:, 0] - cx) / (self.width/2)
        ty = (self.coords[:, 1] - cy) / (self.length/2)
        
        # 비틀림 모사: 1, 3사분면 Up / 2, 4사분면 Down
        loads[:, 2] = 1000.0 * (tx * ty)
        return loads

    def _generate_lifting_pattern(self, corner_idx: int = 0) -> np.ndarray:
        """
        코너 리프팅 하중 패턴 생성.

        Parameters
        ----------
        corner_idx : int
            0=좌하, 1=우하, 2=좌상, 3=우상 코너 (결정적 선택).
        """
        loads = np.zeros((self.num_nodes, 3))
        kx = self.x_min if corner_idx in [0, 2] else self.x_max
        ky = self.y_min if corner_idx in [0, 1] else self.y_max

        dist_sq = (self.coords[:, 0] - kx)**2 + (self.coords[:, 1] - ky)**2
        loads[:, 2] = 1200.0 * np.exp(-dist_sq / (0.1 * (self.width**2)))
        return loads

if __name__ == "__main__":
    # 간단한 테스트 코드
    pass
