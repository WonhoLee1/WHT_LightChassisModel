# -*- coding: utf-8 -*-
"""
wht_dynamic_common.py
=====================
동해석 공통 데이터 구조 및 유틸리티.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from wht_converter.wht_models import WHTMetadata, WHTResultData


# ─────────────────────────────────────────────────────────────────────────────
# 하중 및 감쇠 정의
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DynamicLoadGroup:
    """
    동해석 시각 하중 그룹.

    Attributes
    ----------
    node_ids:   대상 노드 ID 리스트
    dof:        자유도 (0~5: Tx, Ty, Tz, Rx, Ry, Rz)
    magnitude:  진폭 (Force[N] 또는 SPCD[mm])
    time_func:  "half_sine", "step", "ramp", "sine" 등
    t_pulse:    float = 0.02
    t_start:    float = 0.0
    load_type:  str   = "FORCE"
    distribute: bool  = True
    """
    node_ids:   List[int]
    dof:        int
    magnitude:  float
    time_func:  str = "half_sine"
    t_pulse:    float = 0.02
    t_start:    float = 0.0
    load_type:  str   = "FORCE"
    distribute: bool  = True

    def evaluate(self, t: float) -> float:
        """시각 t에서의 하중/변위 스칼라 값 산출."""
        t_eff = t - self.t_start
        if t_eff < 0:
            return 0.0

        amp = self.magnitude
        if self.load_type.upper() == "FORCE" and self.distribute:
            amp /= max(len(self.node_ids), 1)

        if self.time_func == "half_sine":
            if 0 <= t_eff <= self.t_pulse:
                return amp * math.sin(math.pi * t_eff / self.t_pulse)
            return 0.0
        elif self.time_func == "step":
            return amp
        elif self.time_func == "ramp":
            return amp * min(t_eff / self.t_pulse, 1.0)
        elif self.time_func == "sine":
            return amp * math.sin(2.0 * math.pi * t_eff / self.t_pulse)
        return amp

    def u_value(self, t: float) -> float:
        """[SPCD 전용] 처방 변위 u(t)."""
        return self.evaluate(t)

    def ud_value(self, t: float) -> float:
        """[SPCD 전용] 처방 속도 v(t) — d/dt(evaluate)."""
        t_eff = t - self.t_start
        if t_eff < 0 or t_eff > self.t_pulse:
            return 0.0

        amp = self.magnitude
        if self.time_func == "half_sine":
            # d/dt [ A sin(pi t / T) ] = A (pi/T) cos(pi t / T)
            return amp * (math.pi / self.t_pulse) * math.cos(math.pi * t_eff / self.t_pulse)
        elif self.time_func == "sine":
            return amp * (2.0 * math.pi / self.t_pulse) * math.cos(2.0 * math.pi * t_eff / self.t_pulse)
        return 0.0

    def udd_value(self, t: float) -> float:
        """[SPCD 전용] 처방 가속도 ü(t) — d²/dt²(evaluate)."""
        t_eff = t - self.t_start
        if t_eff < 0 or t_eff > self.t_pulse:
            return 0.0

        amp = self.magnitude
        if self.time_func == "half_sine":
            # d²/dt² [ A sin(pi t / T) ] = -A (pi/T)² sin(pi t / T)
            return -amp * (math.pi / self.t_pulse)**2 * math.sin(math.pi * t_eff / self.t_pulse)
        elif self.time_func == "sine":
            # d²/dt² [ A sin(2pi t / T) ] = -A (2pi/T)² sin(2pi t / T)
            return -amp * (2.0 * math.pi / self.t_pulse)**2 * math.sin(2.0 * math.pi * t_eff / self.t_pulse)
        return 0.0


@dataclass
class DampingSpec:
    """감쇠 사양 정의."""
    mode:       str   = "zeta"
    alpha:      float = 0.0
    beta:       float = 0.0
    zeta:       float = 0.02
    omega_ref1: float = 0.0   # rad/s  (0 = 자동)
    omega_ref2: float = 0.0   # rad/s  (0 = 자동)


# ─────────────────────────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────────────────────────

class DynamicResult:
    """동해석 시간 이력 결과."""

    def __init__(self, t_saved: np.ndarray, sorted_nids: List[int]):
        self.t_saved     = t_saved
        self.sorted_nids = sorted_nids
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.solver_type: str  = ""
        self.solver_info: dict = {}
        # 응력/변형률 이력: {"Stress": (T, M, 6), "Von Mises": (T, M), ...}
        self.stress_data: Optional[Dict[str, np.ndarray]] = None

    @property
    def n_save(self) -> int:
        return len(self.t_saved)

    def peak_time_index(self) -> int:
        """전체 최대 변위가 발생하는 저장 인덱스."""
        norms = np.max(np.abs(self.u.reshape(self.n_save, -1)), axis=1)
        return int(np.argmax(norms))

    def to_wht_result_data(self, metadata, mesh_model, load_groups: Optional[List[DynamicLoadGroup]] = None):
        """WHTResultData IR로 변환."""
        from wht_converter.wht_models import WHTResultData

        base_rd = mesh_model.to_wht_result_data(metadata)
        n_nodes = len(self.sorted_nids)
        nid_to_idx = {nid: i for i, nid in enumerate(self.sorted_nids)}

        point_data = {}
        if self.u is not None:
            point_data["Displacement"] = self.u[:, :, :3]
            point_data["Rotation"]     = self.u[:, :, 3:]
        if self.v is not None:
            point_data["Velocity"]     = self.v[:, :, :3]
        if self.a is not None:
            point_data["Acceleration"] = self.a[:, :, :3]

        if load_groups:
            load_history = np.zeros((self.n_save, n_nodes, 3))
            for i, t in enumerate(self.t_saved):
                for lg in load_groups:
                    if lg.dof > 2: continue
                    val = lg.evaluate(t)
                    for nid in lg.node_ids:
                        idx = nid_to_idx.get(nid)
                        if idx is not None:
                            load_history[i, idx, lg.dof] += val
            point_data["Applied_Load"] = load_history

        # [WHT] Include SPCD nodes in SPC set for visualization markers

        if load_groups:
            spcd_nids_all = []
            for lg in load_groups:
                if lg.load_type.upper() == "SPCD":
                    spcd_nids_all.extend(lg.node_ids)
            if spcd_nids_all:
                spcd_nids_unique = sorted(list(set(spcd_nids_all)))
                spcd_indices = np.array([nid_to_idx[n] for n in spcd_nids_unique if n in nid_to_idx], dtype=np.int64)
                
                if "SPC" in base_rd.node_sets:
                    existing = base_rd.node_sets["SPC"]
                    base_rd.node_sets["SPC"] = np.unique(np.concatenate([existing, spcd_indices]))
                else:
                    base_rd.node_sets["SPC"] = spcd_indices

        cell_data = self._build_cell_data(mesh_model)
        
        # [WHT] Nodal Stress Recovery 지원: stress_data 배열의 1번째 차원이 노드 수와 같으면 PointData로 매핑
        if self.stress_data is not None:
            for k, arr in list(cell_data.items()):
                if arr.shape[1] == n_nodes:
                    point_data[k] = arr
                    del cell_data[k]
                elif self.stress_data[k].shape[1] == n_nodes:
                    # _build_cell_data가 M_mesh로 zero-pad를 시도했을 수 있으므로 원본을 사용
                    point_data[k] = self.stress_data[k]
                    del cell_data[k]

        return WHTResultData(
            nodes        = base_rd.nodes,
            connectivity = base_rd.connectivity,
            offsets      = base_rd.offsets,
            cell_types   = base_rd.cell_types,
            node_sets    = base_rd.node_sets,
            element_sets = base_rd.element_sets,
            point_data   = point_data,
            cell_data    = cell_data,
            field_data   = {},
            time_values  = self.t_saved,
            metadata     = metadata,
        )

    def _build_cell_data(self, mesh_model=None) -> Dict[str, np.ndarray]:
        """stress_data로부터 visualizer용 cell_data 딕셔너리를 구성합니다.

        WHTResultData 검증 규칙상 cell_data[:, shape[1]] == M_total_cells 이어야 하므로,
        stress/strain 배열(요소만)을 메시 전체 셀 수에 맞게 zero-pad 합니다.

        Parameters
        ----------
        mesh_model : WHTMeshModel, optional
            메시 모델. 제공 시 전체 셀 수(M_mesh)를 계산하여 패딩 수행.

        Returns
        -------
        dict
            Keys: 응력/변형률 필드명, Values: (T, M_mesh, ...) ndarray
        """
        cell_data: Dict[str, np.ndarray] = {}
        if self.stress_data is None:
            return cell_data

        # M_mesh: 메시의 전체 셀 수(쉘 + 빔 + RBE 가상 라인 등 포함)
        M_mesh = None
        if mesh_model is not None:
            M_mesh = len(mesh_model.elements)
            # RBE2/RBE3는 각 Slave Node마다 가상 라인 요소를 생성함 (wht_mesh_model.py 참조)
            for rbe in list(mesh_model.rbe2s.values()) + list(mesh_model.rbe3s.values()):
                M_mesh += len(rbe.slave_nids)

        n_nodes = len(self.sorted_nids) if hasattr(self, 'sorted_nids') else -1
        for key, arr in list(self.stress_data.items()):
            if arr.shape[1] == n_nodes:
                cell_data[key] = arr
                continue
            # arr.shape: (T, M_elem, 6) 또는 (T, M_elem)
            if M_mesh is not None and arr.shape[1] != M_mesh:
                T = arr.shape[0]
                if arr.ndim == 3:
                    padded = np.zeros((T, M_mesh, arr.shape[2]), dtype=arr.dtype)
                    padded[:, :arr.shape[1], :] = arr
                else:
                    padded = np.zeros((T, M_mesh), dtype=arr.dtype)
                    padded[:, :arr.shape[1]] = arr
                cell_data[key] = padded
            else:
                cell_data[key] = arr

        return cell_data

    def summary(self) -> str:
        peak_i = self.peak_time_index()
        u_peak = float(np.max(np.abs(self.u[peak_i])))
        return (
            f"DynamicResult [{self.solver_type}] "
            f"n_save={self.n_save}, T={self.t_saved[-1]:.3f}s, "
            f"peak_disp={u_peak:.3e} mm @ t={self.t_saved[peak_i]:.4f}s"
        )


def assemble_rayleigh_C(alpha: float, beta: float, M_diag: np.ndarray, K_sparse) -> any:
    from scipy.sparse import diags
    M_mat = diags([M_diag], [0], format="csr")
    return alpha * M_mat + beta * K_sparse


def damping_from_zeta(zeta: float, omega1: float, omega2: float) -> tuple[float, float]:
    A = 0.5 * np.array([[1.0 / omega1, omega1], [1.0 / omega2, omega2]])
    b = np.array([zeta, zeta])
    alpha, beta = np.linalg.solve(A, b)
    return float(alpha), float(beta)


def newmark_coeffs(beta: float, gamma: float, dt: float) -> dict:
    a0 = 1.0 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    return dict(a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5)
