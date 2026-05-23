# -*- coding: utf-8 -*-
"""
wht_dynamic_utils.py
====================
CSV 실측 데이터 처리 및 동적 하중 추출을 위한 유틸리티 함수 모음.

CSV 형식
--------
  # chassis, width, length, height
  # start_time, 0.0
  # C1, x_mm, y_mm, z_mm
  ...
  # C8, x_mm, y_mm, z_mm
  Frame,Time,C1_X,C1_Y,C1_Z,...,C8_X,C8_Y,C8_Z
  0,0.0,...
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(y: np.ndarray, box_pts: int = 5) -> np.ndarray:
    """이동 평균 스무딩 (끝단 보호)."""
    box = np.ones(box_pts) / box_pts
    y_s = np.convolve(y, box, mode='same')
    y_s[:box_pts]  = y[:box_pts]
    y_s[-box_pts:] = y[-box_pts:]
    return y_s


def _diff1(y: np.ndarray, dt: float) -> np.ndarray:
    """스무딩 → gradient → 스무딩으로 1차 미분."""
    return _smooth(np.gradient(_smooth(y), dt))


def _diff2(y: np.ndarray, dt: float) -> np.ndarray:
    """스무딩 → gradient → 스무딩 → gradient → 스무딩으로 2차 미분."""
    return _smooth(np.gradient(_diff1(y, dt), dt))


# ─────────────────────────────────────────────────────────────────────────────
# CSV 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv_header(csv_path: str) -> Dict:
    """
    CSV 파일의 # 주석 헤더에서 코너 기준 좌표(mm)와 start_time(s)을 파싱합니다.

    Returns
    -------
    dict
        'corner_positions': {name: (x_mm, y_mm, z_mm), ...}  — C1~C8
        'start_time'      : float | None
    """
    corner_positions: Dict[str, Tuple[float, float, float]] = {}
    start_time: Optional[float] = None
    with open(Path(csv_path), encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith('#'):
                break
            parts = [p.strip() for p in stripped[1:].split(',')]
            if not parts:
                continue
            key = parts[0].lower().replace(' ', '_')
            if key == 'start_time' and len(parts) >= 2:
                start_time = float(parts[1])
            elif re.fullmatch(r'C[1-8]', parts[0]) and len(parts) >= 4:
                corner_positions[parts[0]] = (
                    float(parts[1]), float(parts[2]), float(parts[3])
                )
    return {'corner_positions': corner_positions, 'start_time': start_time}


# ─────────────────────────────────────────────────────────────────────────────
# FEM 노드 탐색
# ─────────────────────────────────────────────────────────────────────────────

def find_nodes_for_corners(
    node_db: Dict[int, List[float]],
    corner_positions: Dict[str, Tuple[float, float, float]],
    n_nodes: int = 3,
    z_face: Optional[float] = None,
    z_tol: float = 2.0,
) -> Dict[str, List[int]]:
    """
    각 코너 기준 좌표에서 가장 가까운 FEM 노드를 n_nodes개씩 탐색합니다.
    전역적으로 중복 없이 그리디 할당 (거리 오름차순 처리).

    CSV 헤더의 코너 좌표는 세계좌표 (낙하 높이, 임의 원점 오프셋 포함)이므로
    FEM 모델 좌표계와 직접 비교할 수 없습니다.
    코너들의 **상대적 XY 배치**(+X+Y, -X-Y 등)만 보존하여 FEM 메시 좌표계로
    정규화 매핑한 후 탐색합니다.

    Parameters
    ----------
    z_face : FEM 바닥면 Z [mm]. None이면 메시 최소 Z를 자동 탐지.
    z_tol  : 바닥면 Z 허용 오차 [mm] (기본: 2.0)
    """
    nid_arr = np.array(list(node_db.keys()), dtype=int)
    xyz_arr = np.array(list(node_db.values()))

    # ── 바닥면 노드 필터 ───────────────────────────────────────────────────
    if z_face is None:
        z_face = float(xyz_arr[:, 2].min())
    face_mask = np.abs(xyz_arr[:, 2] - z_face) <= z_tol
    if face_mask.sum() < n_nodes:
        face_mask = np.ones(len(nid_arr), dtype=bool)
    nid_face = nid_arr[face_mask]
    xyz_face = xyz_arr[face_mask]

    # ── CSV 코너 XY 좌표 → FEM 메시 좌표계로 정규화 매핑 ─────────────────
    # CSV 코너 원점·스케일이 FEM과 다를 수 있으므로 상대 배치만 보존하여 매핑.
    names = list(corner_positions.keys())
    csv_xy = np.array([(corner_positions[n][0], corner_positions[n][1])
                       for n in names], dtype=float)

    csv_ctr = csv_xy.mean(axis=0)
    csv_c   = csv_xy - csv_ctr                           # 중심화

    fem_ctr = xyz_face[:, :2].mean(axis=0)
    fem_rx  = xyz_face[:, 0].max() - xyz_face[:, 0].min()
    fem_ry  = xyz_face[:, 1].max() - xyz_face[:, 1].min()
    csv_rx  = csv_c[:, 0].max() - csv_c[:, 0].min() + 1e-12
    csv_ry  = csv_c[:, 1].max() - csv_c[:, 1].min() + 1e-12

    # 정규화: CSV의 ±방향 부호 보존, 스케일은 FEM 범위에 맞춤
    csv_mapped = csv_c * np.array([fem_rx / csv_rx, fem_ry / csv_ry]) + fem_ctr

    # ── 탐색 ──────────────────────────────────────────────────────────────
    used: set = set()
    result: Dict[str, List[int]] = {}
    for i, name in enumerate(names):
        tx, ty = csv_mapped[i]
        dists2 = ((xyz_face[:, 0] - tx) ** 2 +
                  (xyz_face[:, 1] - ty) ** 2)
        chosen: List[int] = []
        for idx in np.argsort(dists2):
            nid = int(nid_face[idx])
            if nid not in used:
                chosen.append(nid)
                used.add(nid)
                if len(chosen) == n_nodes:
                    break
        result[name] = chosen
    return result


def find_corner_nodes(node_db: Dict[int, List[float]],
                      width: float, length: float,
                      radius: float, z_min: float, z_max: float,
                      ) -> List[Tuple[Tuple[float, float], List[int]]]:
    """
    4 코너 기준으로 반경 내 + z 범위 노드 그룹 반환.
    순서: C5(+X,+Y), C6(+X,-Y), C7(-X,-Y), C8(-X,+Y)
    """
    all_xyz = np.array([v for v in node_db.values()])
    mask_z  = (all_xyz[:, 2] >= z_min) & (all_xyz[:, 2] <= z_max)
    xyz_z   = all_xyz[mask_z]

    if len(xyz_z) == 0:
        raise RuntimeError(f"z=[{z_min},{z_max}]mm 범위 내 노드가 없습니다.")

    x_min, x_max = xyz_z[:, 0].min(), xyz_z[:, 0].max()
    y_min, y_max = xyz_z[:, 1].min(), xyz_z[:, 1].max()

    targets = [
        (x_max, y_max),  # C5: +X, +Y
        (x_max, y_min),  # C6: +X, -Y
        (x_min, y_min),  # C7: -X, -Y
        (x_min, y_max),  # C8: -X, +Y
    ]

    nid_arr = np.array(list(node_db.keys()))
    xyz_arr = np.array(list(node_db.values()))

    groups = []
    for cx, cy in targets:
        in_z = (xyz_arr[:, 2] >= z_min) & (xyz_arr[:, 2] <= z_max)
        in_r = ((xyz_arr[:, 0] - cx) ** 2 +
                (xyz_arr[:, 1] - cy) ** 2) < radius ** 2
        nids = nid_arr[in_z & in_r].tolist()
        if not nids:
            dists       = (xyz_arr[:, 0] - cx) ** 2 + (xyz_arr[:, 1] - cy) ** 2
            nearest_idx = np.argmin(dists)
            nids        = [int(nid_arr[nearest_idx])]
        groups.append(((cx, cy), nids))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 하중 그룹
# ─────────────────────────────────────────────────────────────────────────────

class InterpLoadGroup:
    """시계열 데이터를 선형 보간하여 하중/변위를 반환하는 하중 그룹."""
    def __init__(self, node_ids: List[int], dof: int, time_arr: np.ndarray,
                 val_arr: np.ndarray, load_type: str = "SPCD"):
        self.node_ids  = node_ids
        self.dof       = dof
        self.load_type = load_type.upper()
        self._t = time_arr
        self._u = val_arr

    def evaluate(self, t: float) -> float:
        return float(np.interp(t, self._t, self._u))

    def u_value(self, t: float) -> float:
        return self.evaluate(t)

    def ud_value(self, t: float) -> float:
        eps = 1e-6
        return (self.evaluate(t + eps) - self.evaluate(t - eps)) / (2 * eps)

    def udd_value(self, t: float) -> float:
        eps = 1e-6
        return (self.evaluate(t + eps) - 2 * self.evaluate(t) +
                self.evaluate(t - eps)) / (eps ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch 기반 강체 제거 + body-frame 변형 추출
# ─────────────────────────────────────────────────────────────────────────────

def calculate_chassis_deformation(
    csv_df,
    time_arr: np.ndarray,
    corner_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    corner_labels: Optional[List[str]] = None,
    contact_accel_threshold: float = 24516.6,
    min_fit_points: int = 3,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kabsch 알고리즘(SVD)으로 강체 병진·회전을 제거하고
    섀시 로컬 프레임 기준 3D 코너 변형량을 반환합니다.

    contact_accel_threshold:
        코너 상대 Z가속도 [mm/s²] 임계값 (기본 2.5g ≈ 24517).
        충격 임펄스 순간에 초과하는 코너는 Kabsch fit에서 제외됩니다.
        정적 지면 접촉처럼 가속도가 작은 경우는 제외되지 않습니다.
    min_fit_points:
        Kabsch fit 최소 포인트 수. 부족 시 직전 R 유지.

    Returns
    -------
    deformation  : {corner_name: (T, 3)} body-frame 변형량 [mm]
    R_arr        : (T, 3, 3) 회전 행렬 (body → world). world = R @ body + T
    T_arr        : (T, 3)   평균 이동 벡터 [mm]. T = Q_mean - R @ P_mean
    contact_mask : (T, n_pts) bool — True = 해당 시간 Kabsch fit 제외
    traj         : (T, n_pts, 3) 세계좌표 위치 [mm]
    vel_arr      : (T, n_pts, 3) 세계좌표 속도 [mm/s]
    accel_arr    : (T, n_pts, 3) 세계좌표 가속도 [mm/s²]
    """
    if corner_labels is None:
        corner_labels = [f'C{i}' for i in range(1, 9)]

    n_steps = len(time_arr)
    n_pts   = len(corner_labels)
    dt      = float(time_arr[1] - time_arr[0]) if n_steps > 1 else 1e-4

    # (T, n_pts, 3) world trajectory [mm]
    traj = np.zeros((n_steps, n_pts, 3))
    for i, lbl in enumerate(corner_labels):
        for j, ax in enumerate(['X', 'Y', 'Z']):
            col = (f"{lbl}_{ax}" if f"{lbl}_{ax}" in csv_df.columns
                   else f"{lbl}_pos_{ax}")
            if col in csv_df.columns:
                traj[:, i, j] = csv_df[col].to_numpy(dtype=float) * 1000.0

    # 스무딩 후 속도·가속도 (세계좌표)
    vel_arr   = np.zeros_like(traj)
    accel_arr = np.zeros_like(traj)
    for i in range(n_pts):
        for j in range(3):
            vel_arr[:, i, j]   = _diff1(traj[:, i, j], dt)
            accel_arr[:, i, j] = _diff2(traj[:, i, j], dt)

    # 충격 임펄스 감지: 코너 상대 Z가속도 > threshold
    mean_az      = accel_arr[:, :, 2].mean(axis=1, keepdims=True)  # (T, 1)
    rel_az       = accel_arr[:, :, 2] - mean_az                    # (T, n_pts)
    contact_mask = np.abs(rel_az) > contact_accel_threshold        # (T, n_pts)

    # 기준 형상 P_ref
    if corner_positions is not None:
        P_ref = np.array([corner_positions.get(lbl, (0.0, 0.0, 0.0))
                          for lbl in corner_labels], dtype=float)
    else:
        P_ref = traj[0].copy()

    P_mean = P_ref.mean(axis=0)
    P_c    = P_ref - P_mean

    R_arr    = np.zeros((n_steps, 3, 3))
    T_arr    = np.zeros((n_steps, 3))
    body_pos = np.zeros((n_steps, n_pts, 3))
    R_prev   = np.eye(3)

    for t in range(n_steps):
        Q      = traj[t]
        Q_mean = Q.mean(axis=0)
        Q_c    = Q - Q_mean

        free = ~contact_mask[t]
        if free.sum() >= min_fit_points:
            H         = Q_c[free].T @ P_c[free]
            U, _S, Vt = np.linalg.svd(H)
            d         = np.linalg.det(Vt.T @ U.T)
            D         = np.diag([1.0, 1.0, float(np.sign(d))])
            R         = Vt.T @ D @ U.T
            R_prev    = R
        else:
            R = R_prev   # fallback: 직전 R 유지

        R_arr[t]    = R
        T_arr[t]    = Q_mean - R @ P_mean
        body_pos[t] = Q_c @ R.T + P_mean   # (n_pts, 3)

    body_pos_0 = body_pos[0].copy()
    deformation: Dict[str, np.ndarray] = {}
    for i, lbl in enumerate(corner_labels):
        deformation[lbl] = body_pos[:, i, :] - body_pos_0[i]   # (T, 3)

    return deformation, R_arr, T_arr, contact_mask, traj, vel_arr, accel_arr


# ─────────────────────────────────────────────────────────────────────────────
# 진단 그래프 저장
# ─────────────────────────────────────────────────────────────────────────────

_CORNER_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]


def plot_kabsch_diagnostics(
    time_arr: np.ndarray,
    traj: np.ndarray,
    vel_arr: np.ndarray,
    accel_arr: np.ndarray,
    deformation: Dict[str, np.ndarray],
    contact_mask: np.ndarray,
    contact_threshold: float,
    corner_labels: List[str],
    save_paths: List[str],
    loadcase_name: str = "",
) -> None:
    """
    Kabsch 전처리 진단 그래프를 PNG로 저장합니다.

    방향(X/Y/Z)별 3개 figure, 각각 7행:
      1. 세계좌표 위치 [mm]
      2. 세계좌표 속도 [mm/s]
      3. 세계좌표 상대가속도 [g]  (Z만 임계값 표시)
      4. body-frame 변형 변위 [mm]
      5. body-frame 변형 속도 [mm/s]   ← 변위 미분
      6. body-frame 변형 가속도 [mm/s²] ← 변위 2차 미분
      7. Kabsch 포함 여부 (코너별 step)

    Parameters
    ----------
    save_paths : 기본 경로 목록. 방향 suffix (_X/Y/Z.png)가 자동 부가됩니다.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_pts  = len(corner_labels)
    colors = _CORNER_COLORS[:n_pts]
    g_mm   = 9806.65   # 1g in mm/s²
    dt     = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1e-4

    # body-frame 변형 속도·가속도 (변위에서 안정 미분)
    body_vel:   Dict[str, np.ndarray] = {}
    body_accel: Dict[str, np.ndarray] = {}
    for lbl in corner_labels:
        d = deformation.get(lbl)
        if d is not None:
            body_vel[lbl]   = np.stack([_diff1(d[:, j], dt) for j in range(3)], axis=1)
            body_accel[lbl] = np.stack([_diff2(d[:, j], dt) for j in range(3)], axis=1)

    # 충격 구간 (contact_mask 기준) → 배경 음영용
    any_contact = contact_mask.any(axis=1)
    transitions = np.diff(any_contact.astype(int))
    span_starts = list(np.where(transitions == 1)[0])
    span_ends   = list(np.where(transitions == -1)[0])
    if any_contact[0]:
        span_starts = [0] + span_starts
    if any_contact[-1]:
        span_ends   = span_ends + [len(time_arr) - 1]

    def _shade(ax_list):
        for s, e in zip(span_starts, span_ends):
            for a in ax_list:
                a.axvspan(time_arr[s], time_arr[e], alpha=0.07, color='red')

    row_labels = [
        f'World pos [mm]',
        f'World vel [mm/s]',
        f'Rel accel [g]',
        f'Body deform disp [mm]',
        f'Body deform vel [mm/s]',
        f'Body deform accel [mm/s2]',
        f'Kabsch included',
    ]

    for ax_idx, ax_name in enumerate(['X', 'Y', 'Z']):
        fig, axes = plt.subplots(7, 1, figsize=(7, 12), sharex=True)
        fig.suptitle(
            f"Kabsch Diagnostics — {loadcase_name}  [{ax_name}]",
            fontsize=13, fontweight='bold',
        )

        # 1. 세계좌표 위치
        for i, lbl in enumerate(corner_labels):
            axes[0].plot(time_arr, traj[:, i, ax_idx],
                         color=colors[i], linewidth=0.8, label=lbl)
        axes[0].legend(loc='upper right', ncol=4, fontsize=7)

        # 2. 세계좌표 속도
        for i, lbl in enumerate(corner_labels):
            axes[1].plot(time_arr, vel_arr[:, i, ax_idx],
                         color=colors[i], linewidth=0.8)

        # 3. 상대 가속도 + 임계값 (Z만)
        mean_a = accel_arr[:, :, ax_idx].mean(axis=1)
        for i, lbl in enumerate(corner_labels):
            axes[2].plot(time_arr,
                         (accel_arr[:, i, ax_idx] - mean_a) / g_mm,
                         color=colors[i], linewidth=0.8)
        if ax_idx == 2:
            thr_g = contact_threshold / g_mm
            axes[2].axhline( thr_g, color='red', linestyle='--',
                             linewidth=1.0, label=f'+{thr_g:.1f}g')
            axes[2].axhline(-thr_g, color='red', linestyle='--', linewidth=1.0)
            axes[2].legend(loc='upper right', fontsize=8)

        # 4. body-frame 변형 변위
        for i, lbl in enumerate(corner_labels):
            d = deformation.get(lbl)
            if d is not None:
                axes[3].plot(time_arr, d[:, ax_idx],
                             color=colors[i], linewidth=0.8)
        axes[3].axhline(0, color='grey', linewidth=0.5, linestyle=':')

        # 5. body-frame 변형 속도
        for i, lbl in enumerate(corner_labels):
            bv = body_vel.get(lbl)
            if bv is not None:
                axes[4].plot(time_arr, bv[:, ax_idx],
                             color=colors[i], linewidth=0.8)
        axes[4].axhline(0, color='grey', linewidth=0.5, linestyle=':')

        # 6. body-frame 변형 가속도
        for i, lbl in enumerate(corner_labels):
            ba = body_accel.get(lbl)
            if ba is not None:
                axes[5].plot(time_arr, ba[:, ax_idx],
                             color=colors[i], linewidth=0.8)
        axes[5].axhline(0, color='grey', linewidth=0.5, linestyle=':')

        # 7. Kabsch 포함 여부 (코너별 step, 0.1 오프셋)
        for i, lbl in enumerate(corner_labels):
            included = (~contact_mask[:, i]).astype(float)
            axes[6].step(time_arr, included + i * 1.2,
                         color=colors[i], linewidth=0.9, where='post')
        axes[6].set_yticks([i * 1.2 + 0.5 for i in range(n_pts)])
        axes[6].set_yticklabels(corner_labels, fontsize=7)
        axes[6].set_ylim(-0.2, n_pts * 1.2)
        axes[6].grid(True, axis='x', linewidth=0.4)

        for idx, ax in enumerate(axes):
            ax.set_ylabel(row_labels[idx].replace(
                'pos', ax_name).replace('vel', f'V{ax_name}')
                .replace('accel', f'A{ax_name}')
                .replace('disp', ax_name), fontsize=8)
            ax.grid(True, linewidth=0.4)

        axes[6].set_xlabel('Time [s]')
        _shade(axes)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for base_path in save_paths:
            png = Path(base_path).parent / f"{Path(base_path).stem}_{ax_name}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(png), dpi=120, bbox_inches='tight')

        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 전처리 래퍼: 자세별 최초 1회 실행 + 진단 저장
# ─────────────────────────────────────────────────────────────────────────────

class KabschPreprocessor:
    """Kabsch 기반 강체 제거 전처리기.

    fit() 호출 후 R_arr, T_arr, deformation 등을 멤버 변수로 보관합니다.
    시각화 시 apply_rigid_body()로 body-frame → world-frame 복원이 가능합니다.
    """

    def __init__(
        self,
        contact_accel_threshold: float = 24516.6,
        min_fit_points: int = 3,
    ) -> None:
        self.contact_accel_threshold = contact_accel_threshold
        self.min_fit_points = min_fit_points

        self.deformation:  Dict[str, np.ndarray] = {}
        self.R_arr:        Optional[np.ndarray]  = None  # (T, 3, 3) 회전행렬 (body→world)
        self.T_arr:        Optional[np.ndarray]  = None  # (T, 3) 평균 이동벡터 [mm] (world frame)
        self.contact_mask: Optional[np.ndarray]  = None  # (T, n_pts) bool
        self.traj:         Optional[np.ndarray]  = None  # (T, n_pts, 3) 세계좌표 위치
        self.vel_arr:      Optional[np.ndarray]  = None
        self.accel_arr:    Optional[np.ndarray]  = None
        self.time_arr:     Optional[np.ndarray]  = None  # (T,) 0-based 시간 [s]
        self._corner_labels: List[str]           = []

    def fit(
        self,
        csv_df,
        time_arr: np.ndarray,
        corner_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        corner_labels: Optional[List[str]] = None,
        diag_save_paths: Optional[List[str]] = None,
        loadcase_name: str = "",
    ) -> "KabschPreprocessor":
        """Kabsch 전처리를 실행하고 결과를 멤버 변수에 저장합니다."""
        if corner_labels is None:
            corner_labels = [f'C{i}' for i in range(1, 9)]
        self._corner_labels = corner_labels
        self.time_arr = time_arr

        thr_g = self.contact_accel_threshold / 9806.65
        print(f"\n [Kabsch] {loadcase_name}  "
              f"threshold={thr_g:.1f}g  corners={len(corner_labels)}")

        (self.deformation, self.R_arr, self.T_arr,
         self.contact_mask, self.traj, self.vel_arr, self.accel_arr) = \
            calculate_chassis_deformation(
                csv_df, time_arr,
                corner_positions=corner_positions,
                corner_labels=corner_labels,
                contact_accel_threshold=self.contact_accel_threshold,
                min_fit_points=self.min_fit_points,
            )

        n_steps = len(time_arr)
        print(f"       {'corner':<6}  {'excl frames':>11}  {'excl%':>6}"
              f"  {'|dX|pk':>8}  {'|dY|pk':>8}  {'|dZ|pk':>8}  mm")
        for i, lbl in enumerate(corner_labels):
            n_exc = int(self.contact_mask[:, i].sum())
            d     = self.deformation[lbl]
            print(f"       {lbl:<6}  {n_exc:>11}  {n_exc/n_steps*100:>5.1f}%"
                  f"  {np.max(np.abs(d[:,0])):>8.3f}"
                  f"  {np.max(np.abs(d[:,1])):>8.3f}"
                  f"  {np.max(np.abs(d[:,2])):>8.3f}")

        if diag_save_paths:
            plot_kabsch_diagnostics(
                time_arr, self.traj, self.vel_arr, self.accel_arr,
                self.deformation, self.contact_mask,
                self.contact_accel_threshold, corner_labels,
                diag_save_paths, loadcase_name,
            )
            for base_path in diag_save_paths:
                for ax_name in ('X', 'Y', 'Z'):
                    png = Path(base_path).parent / f"{Path(base_path).stem}_{ax_name}.png"
                    if png.exists():
                        print(f"       saved: {png}")

        return self

    def apply_rigid_body(
        self,
        base_pts: np.ndarray,
        u_fem: np.ndarray,
        t_idx: int,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Body-frame 좌표를 world-frame으로 복원합니다 (시각화용).

        x_vis = R(t) @ (base_pts + u_fem * scale) + T(t)

        Parameters
        ----------
        base_pts : (N, 3) 기준 노드 좌표
        u_fem    : (N, 3) FEM 변위 (body-frame 소변형)
        t_idx    : 시간 스텝 인덱스
        scale    : 변위 배율 (변형 과장 표시용)
        """
        if self.R_arr is None or self.T_arr is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        t_idx = min(t_idx, len(self.R_arr) - 1)
        R = self.R_arr[t_idx]   # (3, 3)
        T = self.T_arr[t_idx]   # (3,)
        deformed = base_pts + u_fem * scale
        return (R @ deformed.T).T + T


def run_kabsch_preprocessing(
    csv_df,
    time_arr: np.ndarray,
    corner_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    corner_labels: Optional[List[str]] = None,
    contact_accel_threshold: float = 24516.6,
    min_fit_points: int = 3,
    diag_save_paths: Optional[List[str]] = None,
    loadcase_name: str = "",
) -> "KabschPreprocessor":
    """KabschPreprocessor.fit()의 하위 호환 래퍼.

    Returns
    -------
    KabschPreprocessor  (deformation, R_arr, T_arr 등 멤버 변수로 접근)
    """
    return KabschPreprocessor(
        contact_accel_threshold=contact_accel_threshold,
        min_fit_points=min_fit_points,
    ).fit(
        csv_df, time_arr,
        corner_positions=corner_positions,
        corner_labels=corner_labels,
        diag_save_paths=diag_save_paths,
        loadcase_name=loadcase_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 레거시: 4코너 cross-product 기반 로컬 Z 추출 (하위 호환)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_local_z_history(
    csv_df, time_arr: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    4개 코너(C5~C8) 궤적에서 강체 회전 제거 후 로컬 Z 변위를 추출합니다.

    .. deprecated::
        Face-down 등 비수평 낙하에서 로컬 프레임이 퇴화합니다.
        run_kabsch_preprocessing 사용을 권장합니다.
    """
    n_steps       = len(time_arr)
    corner_labels = ['C5', 'C6', 'C7', 'C8']

    traj = np.zeros((n_steps, 4, 3))
    for i, lbl in enumerate(corner_labels):
        for j, ax in enumerate(['X', 'Y', 'Z']):
            col = (f"{lbl}_{ax}" if f"{lbl}_{ax}" in csv_df.columns
                   else f"{lbl}_pos_{ax}")
            if col in csv_df.columns:
                traj[:, i, j] = csv_df[col].to_numpy(dtype=float) * 1000.0

    local_z_results = {lbl: np.zeros(n_steps) for lbl in corner_labels}
    p_loc_t0 = None

    for t in range(n_steps):
        pts    = traj[t]
        origin = np.mean(pts, axis=0)
        p_c    = pts - origin

        v_x = ((p_c[1] - p_c[2]) + (p_c[0] - p_c[3])) / 2.0
        v_y = ((p_c[3] - p_c[2]) + (p_c[0] - p_c[1])) / 2.0

        z_loc  = np.cross(v_x, v_y)
        z_norm = np.linalg.norm(z_loc)
        if z_norm < 1e-12:
            z_loc = np.array([0, 0, 1.0])
        else:
            z_loc /= z_norm

        if z_loc @ np.array([0, 0, 1.0]) < 0:
            z_loc = -z_loc

        x_loc = v_x / (np.linalg.norm(v_x) + 1e-12)
        y_loc = np.cross(z_loc, x_loc)
        x_loc = np.cross(y_loc, z_loc)

        R     = np.stack([x_loc, y_loc, z_loc], axis=1)
        p_loc = p_c @ R

        if t == 0:
            p_loc_t0 = p_loc.copy()

        delta_p_loc = p_loc - p_loc_t0
        for i, lbl in enumerate(corner_labels):
            local_z_results[lbl][t] = delta_p_loc[i, 2]

    return local_z_results, traj


def calculate_corner_accelerations(traj: np.ndarray, dt: float) -> np.ndarray:
    """(T, 4, 3) 궤적에서 4코너 각각의 Z 가속도를 산출합니다."""
    n_steps = traj.shape[0]
    accels  = np.zeros((n_steps, 4))
    for i in range(4):
        accels[:, i] = _diff2(traj[:, i, 2], dt)
    return accels


def compute_vk_inertia_scale(
    E: float, nu: float, t: float,
    plate_a: float, plate_b: float,
    total_mass: float, a_body_z: np.ndarray,
    g_mm_s2: float = 9806.65,
) -> tuple:
    """
    Von Kármán 대변형 이론 기반 관성력 보정계수를 계산합니다.

    단순지지 사각판(면내 고정)을 기준 평판으로 사용하여 선형/비선형 처짐 비를
    구하고, 이를 관성력 스케일 팩터로 반환합니다.

    [이론 배경]
    Timoshenko 근사식 (단순지지 균일 하중 사각판):
      q * a^4 / (D * t) = A * (w/t) + B * (w/t)^3
    - A = 49.7  : 굽힘 항 (선형)
    - B = 17.1  : 막 작용 항 (비선형, 면내 고정 경계)
    - 선형이론: B 항 무시 → w_L = (lhs/A) * t
    - 비선형이론: 3차 방정식 수치 풀이 → w_NL
    - 보정계수 α = w_NL / w_L  (항상 < 1)

    [주의]
    실제 모델은 코너 4점 지지이므로 면내 구속이 없어 막 작용이 이론치보다 약합니다.
    이 보정계수는 보수적 상한 보정(막 작용 과대 반영)을 제공합니다.

    Parameters
    ----------
    E, nu, t     : 재료 탄성계수, 포아송비, 두께 [MPa, -, mm]
    plate_a, b   : 평판 치수 (단변/장변) [mm]
    total_mass   : 총 구조 질량 [tonne]
    a_body_z     : (T,) body-frame Z 강체 가속도 [mm/s²]
    g_mm_s2      : 중력가속도 [mm/s²]

    Returns
    -------
    scale        : float  보정계수 (w_NL / w_L)
    w_L          : float  선형이론 최대 처짐 [mm]
    w_NL         : float  비선형이론 최대 처짐 [mm]
    info         : dict   진단 정보
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    a = min(plate_a, plate_b)           # 단변 기준
    area = plate_a * plate_b

    a_peak = float(np.max(np.abs(a_body_z)))
    F_peak = total_mass * a_peak        # N (tonne * mm/s^2 = N)
    q      = F_peak / area              # N/mm^2

    A_coef, B_coef = 49.7, 17.1
    lhs = q * a**4 / (D * t)

    # 선형이론
    wt_L = lhs / A_coef
    w_L  = wt_L * t

    # 비선형이론: B*x^3 + A*x - lhs = 0
    roots = np.roots([B_coef, 0.0, A_coef, -lhs])
    real_pos = sorted(
        [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    )
    if real_pos:
        wt_NL = real_pos[0]
    else:
        wt_NL = wt_L  # fallback: 보정 없음

    w_NL  = wt_NL * t
    scale = w_NL / w_L if w_L > 0 else 1.0

    info = dict(
        D=D, a_ref=a, area=area,
        q=q, F_peak=F_peak,
        a_peak_g=a_peak / g_mm_s2,
        lhs=lhs,
        w_L=w_L, wt_L=wt_L,
        w_NL=w_NL, wt_NL=wt_NL,
        scale=scale,
    )
    return scale, w_L, w_NL, info


def print_vk_scale_report(info: dict) -> None:
    """compute_vk_inertia_scale 결과를 터미널에 출력합니다."""
    sep = "     " + "-" * 58
    print(sep)
    print("     [Von Karman 비선형 관성력 보정]")
    print(sep)
    print(f"     기준 평판 : 단순지지 사각판 (면내 고정), 단변 {info['a_ref']:.0f} mm")
    print(f"     굽힘 강성 : D = {info['D']:,.1f} N*mm")
    print(f"     등가 압력 : q = {info['q']:.4e} N/mm^2  "
          f"(peak F = {info['F_peak']:,.0f} N  @ {info['a_peak_g']:.1f} g)")
    print(f"     무차원 하중: q*a^4/(D*t) = {info['lhs']:.1f}")
    print(sep)
    print(f"     선형 이론 : w/t = {info['wt_L']:>8.1f}   w = {info['w_L']:>8.1f} mm")
    print(f"     비선형 이론: w/t = {info['wt_NL']:>8.2f}   w = {info['w_NL']:>8.2f} mm")
    print(sep)
    print(f"     보정계수 alpha = w_NL / w_L = {info['scale']:.5f}")
    print(f"     -> 관성력을 {info['scale']*100:.2f}% 수준으로 스케일 적용")
    print(f"     [주의] 코너 4점 지지는 면내 구속 없음 -> 막 작용 과대 반영")
    print(f"            실제 보정계수는 이보다 클 수 있음 (보수적 적용)")
    print(sep)


def compute_inertia_scale_via_fem(
    model,
    corner_nids: list,
    n2i: dict,
    m_diag: np.ndarray,
    a_body_z: np.ndarray,
    w_NL_target: float,
) -> tuple:
    """
    FEM 정해석으로 관성력 → 구조 변위를 계산하여 Von Kármán 목표 처짐 기준
    보정계수를 역산합니다.

    [절차]
    1. peak 관성력을 정하중으로 구성 (F_z = -m_node * a_peak)
    2. 코너 노드 Z방향 고정 (SPC)
    3. 정해석 실행 → 최대 Z 변위 w_FEM 획득
    4. scale = w_NL_target / w_FEM

    Parameters
    ----------
    model        : WHTMeshModel
    corner_nids  : 코너 슬레이브 노드 목록 (Z SPC 적용 대상)
    n2i          : {nid: idx} 노드 인덱스 맵
    m_diag       : 집중 질량 대각 벡터
    a_body_z     : (T,) body-frame Z 강체 가속도 [mm/s²]
    w_NL_target  : Von Kármán 비선형 목표 처짐 [mm]

    Returns
    -------
    scale        : float  보정계수 (w_NL_target / w_FEM)
    w_FEM        : float  FEM 정해석 최대 Z 변위 [mm]
    """
    import copy
    from wht_solver.wht_solver import WHTSolver
    from wht_solver.load_cases import WHTLoadCase

    a_peak = float(np.max(np.abs(a_body_z)))

    # 임시 모델: 코너 노드 Z 고정
    tmp = copy.deepcopy(model)
    lc  = WHTLoadCase(name="inertia_scale_probe")
    lc.add_bc(corner_nids, dofs=(2,), value=0.0)

    # 모든 구조 노드에 peak 관성력 인가
    for nid in model.nodes:
        if nid >= 900000:
            continue
        idx = n2i.get(nid)
        if idx is None:
            continue
        fm = m_diag[idx * 6 + 2]
        if fm > 1e-12:
            lc.add_force(nid, dofs=(2,), values=(-fm * a_peak,))

    result = WHTSolver(tmp).solve_static(lc)
    u = result.displacement          # (N, 6)
    w_FEM = float(np.max(np.abs(u[:, 2])))

    scale = w_NL_target / w_FEM if w_FEM > 1e-6 else 1.0

    sep = "     " + "-" * 58
    print(sep)
    print("     [FEM 역산 보정 — 코너 Z 고정 정해석]")
    print(f"     peak 관성가속도  : {a_peak/9806.65:.2f} g")
    print(f"     FEM 정해석 최대 Z변위  : w_FEM  = {w_FEM:.2f} mm")
    print(f"     VK 비선형 목표 처짐    : w_NL   = {w_NL_target:.2f} mm")
    print(f"     보정계수 scale = w_NL / w_FEM = {scale:.5f}")
    print(f"     -> 관성력을 {scale*100:.2f}% 수준으로 최종 적용")
    print(sep)

    return scale, w_FEM
