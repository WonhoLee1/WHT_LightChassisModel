import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
from wht_modeler.wht_selector import WHTSelector
from wht_modeler.wht_mesh_model import WHTMeshModel

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

    def get_boundary_nodes(self) -> List[int]:
        """
        WHTSelector를 활용하여 상단 플랜지(고정부) 노드 ID 리스트를 강건하게 식별합니다.
        exam2_shell_jaxSSO_load.py의 로직(box + curvature)을 참고합니다.
        """
        selector = WHTSelector(self.model)
        
        # 1. Bounding Box 기반 최상단 림 선택 (오차범위 0.1mm)
        rim_nids = selector.by_box(z=(self.z_max - 0.1, self.z_max + 0.1)).get_ids()
        
        # 2. 만약 림 노드가 있다면, 인접한 플랜지 면으로 확장 (Curvature 기반)
        if rim_nids:
            # 최상단 노드 중 하나를 시드로 하여 곡률 기반 확장
            # 35.0도 이내의 각도를 가진 인접면 노드들을 선택
            flange_nids = (WHTSelector(self.model)
                           .by_ids(rim_nids)
                           .expand_by_face(angle_limit_deg=35.0, z_min=self.z_max - 5.0)
                           .get_ids())
            return flange_nids
        
        return rim_nids

    def get_load_nodes(self) -> List[int]:
        """
        하중이 주로 인가되는 노드(바닥면 센터 및 코너 등)를 식별하여 반환합니다.
        민감도 해석의 가이드라인으로 활용됩니다.
        """
        # 바닥면(Base Plate) 중앙부 노드 선택
        cx, cy = (self.x_min + self.x_max)/2, (self.y_min + self.y_max)/2
        selector = WHTSelector(self.model)
        
        # 중앙 근처 300mm 반경 노드
        center_nids = selector.by_box(
            x=(cx - 150, cx + 150),
            y=(cy - 150, cy + 150),
            z=(self.z_min - 0.1, self.z_min + 5.0)
        ).get_ids()
        
        return list(center_nids)


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
