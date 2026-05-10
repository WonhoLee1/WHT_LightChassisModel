# -*- coding: utf-8 -*-
"""
wht_dynamic_common.py
=====================
WHT Dynamic Analysis — 공통 데이터 구조 및 유틸리티

DynamicLoadGroup : 노드 그룹별 시간 함수 하중 입력
DampingSpec      : 감쇠 사양 (Rayleigh 또는 모달 감쇠비 ζ)
DynamicResult    : 동해석 시간 이력 저장
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 하중 입력
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DynamicLoadGroup:
    """
    노드 그룹에 대한 시간 의존 하중.

    Parameters
    ----------
    node_ids        : 하중을 적용할 노드 ID 리스트
    dof             : 하중 자유도 (0=Tx, 1=Ty, 2=Tz, 3=Rx, 4=Ry, 5=Rz)
    force_magnitude : 하중 크기 [N 또는 N·mm]
    time_func       : callable(t) → float  또는 프리셋 문자열
                      프리셋: "sine", "half_sine", "step", "ramp", "impulse"
    frequency_hz    : sine/cosine 프리셋 주파수 [Hz]
    t_pulse         : half_sine/ramp 지속 시간 [s]
    distribute      : True이면 force_magnitude를 노드 수로 균등 분배
    """
    node_ids:        List[int]
    dof:             int
    force_magnitude: float
    time_func:       Union[Callable, str] = "step"
    frequency_hz:    float = 0.0
    t_pulse:         float = 0.0
    distribute:      bool  = False

    def evaluate(self, t: float) -> float:
        """시각 t에서의 하중값 반환 (magnitude 반영)."""
        mag = self.force_magnitude
        if self.distribute and len(self.node_ids) > 1:
            mag = mag / len(self.node_ids)

        if callable(self.time_func):
            return mag * float(self.time_func(t))

        name = self.time_func.lower()
        if name == "step":
            return mag
        elif name == "sine":
            return mag * np.sin(2 * np.pi * self.frequency_hz * t)
        elif name == "cosine":
            return mag * np.cos(2 * np.pi * self.frequency_hz * t)
        elif name == "half_sine":
            if self.t_pulse > 0 and 0 <= t <= self.t_pulse:
                return mag * np.sin(np.pi * t / self.t_pulse)
            return 0.0
        elif name == "ramp":
            if self.t_pulse > 0:
                return mag * min(t / self.t_pulse, 1.0)
            return mag
        elif name == "impulse":
            return mag if t == 0.0 else 0.0
        else:
            raise ValueError(
                f"Unknown time_func preset: '{self.time_func}'. "
                "Use: 'step','sine','cosine','half_sine','ramp','impulse' or callable."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 감쇠 사양
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DampingSpec:
    """
    감쇠 사양.

    mode="rayleigh" : C = alpha*M + beta*K  (alpha, beta 직접 지정)
    mode="zeta"     : 임계감쇠비 ζ 지정 → 두 기준 주파수에서 Rayleigh α,β 자동 산출
                      omega_ref1/2 미지정 시 고유값 해석으로 자동 선택.

    Examples
    --------
    DampingSpec(mode="zeta", zeta=0.02)
    DampingSpec(mode="rayleigh", alpha=1.5, beta=2e-4)
    DampingSpec(mode="zeta", zeta=0.05,
                omega_ref1=2*pi*10, omega_ref2=2*pi*100)
    """
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
    """
    동해석 시간 이력 결과.

    Attributes
    ----------
    t_saved     : (n_save,)            저장된 시각 배열 [s]
    sorted_nids : 노드 ID 리스트
    u           : (n_save, n_nodes, 6) 변위 이력 [mm]
    v           : (n_save, n_nodes, 6) 속도 이력 [mm/s]
    a           : (n_save, n_nodes, 6) 가속도 이력 [mm/s²]
    solver_type : "modal" 또는 "direct"
    solver_info : 해석 메타데이터 dict
    """

    def __init__(self, t_saved: np.ndarray, sorted_nids: List[int]):
        self.t_saved     = t_saved
        self.sorted_nids = sorted_nids
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.a: Optional[np.ndarray] = None
        self.solver_type: str  = ""
        self.solver_info: dict = {}

    @property
    def n_save(self) -> int:
        return len(self.t_saved)

    def max_displacement(self) -> np.ndarray:
        """(n_nodes, 6) — 각 DOF의 최대 절대 변위."""
        return np.max(np.abs(self.u), axis=0)

    def peak_time_index(self) -> int:
        """전체 최대 변위가 발생하는 저장 인덱스."""
        norms = np.max(np.abs(self.u.reshape(self.n_save, -1)), axis=1)
        return int(np.argmax(norms))

    def to_wht_result_data(self, metadata, mesh_model):
        """
        WHTResultData IR로 변환 — ParaView 내보내기 및 Visualizer 연동.

        point_data 구성:
            Displacement  : (n_save, n_nodes, 3) [mm]
            Rotation      : (n_save, n_nodes, 3) [rad]
            Velocity      : (n_save, n_nodes, 3) [mm/s]
            Acceleration  : (n_save, n_nodes, 3) [mm/s²]

        Parameters
        ----------
        metadata   : WHTMetadata
        mesh_model : WHTMeshModel (geometry source)

        Returns
        -------
        WHTResultData
        """
        from wht_converter.wht_models import WHTResultData

        base_rd = mesh_model.to_wht_result_data(metadata)

        point_data = {}
        if self.u is not None:
            point_data["Displacement"] = self.u[:, :, :3]
            point_data["Rotation"]     = self.u[:, :, 3:]
        if self.v is not None:
            point_data["Velocity"]     = self.v[:, :, :3]
        if self.a is not None:
            point_data["Acceleration"] = self.a[:, :, :3]

        return WHTResultData(
            nodes        = base_rd.nodes,
            connectivity = base_rd.connectivity,
            offsets      = base_rd.offsets,
            cell_types   = base_rd.cell_types,
            node_sets    = base_rd.node_sets,
            element_sets = base_rd.element_sets,
            point_data   = point_data,
            cell_data    = {},
            field_data   = {},
            time_values  = self.t_saved,
            metadata     = metadata,
        )

    def summary(self) -> str:
        peak_i = self.peak_time_index()
        u_peak = float(np.max(np.abs(self.u[peak_i])))
        return (
            f"DynamicResult [{self.solver_type}] "
            f"n_save={self.n_save}, T={self.t_saved[-1]:.3f}s, "
            f"peak_disp={u_peak:.3e} mm @ t={self.t_saved[peak_i]:.4f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────────────────────────────────────

def assemble_rayleigh_C(
    alpha: float,
    beta: float,
    M_diag: np.ndarray,
    K_sparse,
):
    """
    Rayleigh 비례 감쇠 행렬 조립.

    C = alpha * M + beta * K
    M_diag : (ndof,) 대각 집중 질량 벡터
    K_sparse : (ndof, ndof) scipy sparse 강성 행렬
    """
    from scipy.sparse import diags
    M_mat = diags([alpha * M_diag], [0], format='csr')
    C = M_mat + beta * K_sparse
    return C.tocsr()


def damping_from_zeta(
    zeta: float,
    omega1: float,
    omega2: float,
) -> tuple[float, float]:
    """
    두 기준 주파수(ω₁, ω₂)와 감쇠비 ζ로부터 Rayleigh α, β 산출.

        ζ = α/(2ω) + βω/2  →  2×2 선형계 풀이

    Returns
    -------
    (alpha, beta)
    """
    A = 0.5 * np.array([[1.0 / omega1, omega1],
                         [1.0 / omega2, omega2]])
    b = np.array([zeta, zeta])
    alpha, beta = np.linalg.solve(A, b)
    return float(alpha), float(beta)


def newmark_coeffs(beta: float, gamma: float, dt: float) -> dict:
    """
    Newmark-β 적분 상수 계산.

    beta=0.25, gamma=0.5 : Constant Average Acceleration (무조건 안정)

    Returns
    -------
    dict with keys a0..a5
        a0 = 1/(β dt²)
        a1 = γ/(β dt)
        a2 = 1/(β dt)
        a3 = 1/(2β) - 1
        a4 = γ/β - 1
        a5 = dt(γ/(2β) - 1)
    """
    a0 = 1.0 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    return dict(a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5)
