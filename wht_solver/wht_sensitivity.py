"""
wht_sensitivity.py
==================
WHT FEM Framework — Analytical Sensitivity Analysis

Computes ∂λᵢ/∂p for modal eigenvalues λᵢ with respect to:
  - Element thickness  t_e  (per element, central FD on element K/M)
  - Global Young's modulus E (exact: K ∝ E  →  ∂λ/∂E = λ/E)
  - Global density        ρ (exact: M ∝ ρ  →  ∂λ/∂ρ = -λ/ρ)
  - Nodal topography      z (per node, central FD on global K_free)

Core formula (Rayleigh quotient derivative, M-normalized modes):
    ∂λᵢ/∂p = φᵢᵀ (∂K_free/∂p) φᵢ  −  λᵢ · φᵢᵀ (∂M_free/∂p) φᵢ
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .wht_solver import WHTSolver
    from .wht_result import WHTSolverResult


class WHTSensitivity:
    """
    Analytical sensitivity of modal eigenvalues w.r.t. design variables.

    Usage
    -----
    solver = WHTSolver(model)
    modal  = solver.solve_modal(num_modes=10)
    sens   = WHTSensitivity(solver, modal)

    dλ_dt  = sens.thickness()   # (n_modes, n_elem)
    dλ_dE  = sens.E_global()    # (n_modes,)
    dλ_dρ  = sens.rho_global()  # (n_modes,)
    dλ_dz  = sens.topography()  # (n_modes, n_shell_nodes)  — expensive
    """

    def __init__(self, solver: "WHTSolver", modal_result: "WHTSolverResult"):
        self.solver = solver
        self.modal  = modal_result
        self._build_workspace()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def E_global(self) -> np.ndarray:
        """
        ∂λᵢ/∂E = λᵢ / E

        Exact (K scales linearly with E, M is E-independent).
        """
        return self.lambdas / self._get_E()         # (n_modes,)

    def rho_global(self) -> np.ndarray:
        """
        ∂λᵢ/∂ρ = −λᵢ / ρ

        Exact (M scales linearly with ρ, modes are M-normalised → φᵀMφ = 1).
        """
        return -self.lambdas / self._get_rho()      # (n_modes,)

    def thickness(self, rel_dt: float = 1e-4) -> np.ndarray:
        """
        ∂λᵢ/∂tₑ for every shell element via central FD on element K and M.

        Parameters
        ----------
        rel_dt : relative thickness perturbation  (default 1e-4)

        Returns
        -------
        (n_modes, n_elem) — non-shell elements are zero.
        """
        model      = self.solver.model
        shell_types = ('TRIA3', 'TRIA', 'QUAD4', 'QUAD')
        sorted_eids = sorted(model.elements.keys())
        n_elem      = len(sorted_eids)
        result      = np.zeros((self.n_modes, n_elem))

        for col, eid in enumerate(sorted_eids):
            elem  = model.elements[eid]
            etype = getattr(elem, 'type', '')
            if etype not in shell_types:
                continue

            prop = model.properties.get(elem.pid)
            mat  = model.materials.get(prop.mid) if prop else None
            if not prop or not mat:
                continue

            t, E, nu, rho = prop.t, mat.E, mat.nu, mat.rho
            dt = max(abs(t) * rel_dt, 1e-8)

            crds = [np.array([model.nodes[nid].x,
                              model.nodes[nid].y,
                              model.nodes[nid].z])
                    for nid in elem.node_ids]

            dKe_dt, dMe_dt = self._element_thickness_sens(
                etype, crds, t, E, nu, rho, dt
            )
            result[:, col] = self._scatter_eigensens(
                elem.node_ids, dKe_dt, dMe_dt
            )

        return result

    def topography(
        self,
        dz: float = 0.5,
        node_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        ∂λᵢ/∂zₙ for each shell node via central FD on full K_free.

        Parameters
        ----------
        dz       : z-perturbation in model length units [mm]
        node_ids : nodes to perturb (default: all shell element nodes)

        Returns
        -------
        result   : (n_modes, n_topo_nodes)
        node_ids : the ordered list of node IDs corresponding to columns
        """
        model = self.solver.model
        if node_ids is None:
            shell_types = ('TRIA3', 'TRIA', 'QUAD4', 'QUAD')
            node_set = set()
            for elem in model.elements.values():
                if getattr(elem, 'type', '') in shell_types:
                    node_set.update(elem.node_ids)
            node_ids = sorted(node_set)

        n_topo = len(node_ids)
        result  = np.zeros((self.n_modes, n_topo))

        print(f"    [Topo sens] Perturbing {n_topo} nodes (±{dz} mm)...", flush=True)
        for col, nid in enumerate(node_ids):
            if col % 200 == 0:
                print(f"      {col}/{n_topo}", end='\r', flush=True)

            node   = model.nodes[nid]
            z_orig = node.z

            node.z = z_orig + dz
            K_plus, *_ = self.solver._prepare_matrices()

            node.z = z_orig - dz
            K_minus, *_ = self.solver._prepare_matrices()

            node.z = z_orig          # restore immediately

            dK_dz = (K_plus - K_minus) / (2.0 * dz)  # sparse

            for i in range(self.n_modes):
                phi = self.vecs_free[:, i]
                result[i, col] = phi @ (dK_dz @ phi)   # ∂M/∂z ≈ 0

        print(f"      Done.                    ", flush=True)
        return result, node_ids

    # ------------------------------------------------------------------
    # Workspace construction
    # ------------------------------------------------------------------

    def _build_workspace(self):
        """Compute K_free, M_free, and free-DOF mode shapes once."""
        (K_free, M_free, _jm_ndof,
         unknown_id, sorted_nids, nid_to_idx) = self.solver._prepare_matrices()

        self.K_free      = K_free          # (ndof_free, ndof_free) CSC
        self.M_free      = M_free          # (ndof_free,) diagonal
        self.unknown_id  = unknown_id      # (ndof_free,) global DOF indices
        self.sorted_nids = sorted_nids
        self.nid_to_idx  = nid_to_idx
        # O(1) lookup: global DOF → free-DOF row index
        self.global_to_free: Dict[int, int] = {
            int(g): f for f, g in enumerate(unknown_id)
        }

        # Reconstruct vecs_free (ndof_free, n_modes) from mode_shapes
        # mode_shapes: (n_modes, n_nodes, 6)
        # vecs_full[i*6 : i*6+6, m] = mode_shapes[m, i, :]  (idx == i for sorted)
        ms      = self.modal.mode_shapes              # (n_modes, n_nodes, 6)
        n_modes = ms.shape[0]
        n_nodes = ms.shape[1]
        vecs_full       = ms.transpose(1, 2, 0).reshape(n_nodes * 6, n_modes)
        self.vecs_free  = vecs_full[unknown_id, :]    # (ndof_free, n_modes)

        # ω² = (2πf)²
        self.lambdas = (2.0 * np.pi * self.modal.frequencies) ** 2
        self.n_modes = n_modes

    # ------------------------------------------------------------------
    # Material helpers
    # ------------------------------------------------------------------

    def _get_E(self) -> float:
        for mat in self.solver.model.materials.values():
            return float(mat.E)
        return 210000.0

    def _get_rho(self) -> float:
        for mat in self.solver.model.materials.values():
            return float(mat.rho)
        return 7.85e-9

    # ------------------------------------------------------------------
    # Element-level thickness sensitivity
    # ------------------------------------------------------------------

    def _element_thickness_sens(
        self,
        etype: str,
        crds: List[np.ndarray],
        t: float,
        E: float,
        nu: float,
        rho: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Central FD for ∂Ke/∂t and analytical ∂Me/∂t.

        Returns
        -------
        dKe_dt : (n_dof_e, n_dof_e)
        dMe_dt : (n_dof_e,)  — only translational/rotational diagonal terms
        """
        from .wht_tria3_element import _element_K_tria3, tri_area
        from .wht_quad4_element import _element_K_mitc4_plus

        if etype in ('TRIA3', 'TRIA'):
            c1, c2, c3 = crds
            Kp = _element_K_tria3(c1, c2, c3, t + dt, E, nu)
            Km = _element_K_tria3(c1, c2, c3, t - dt, E, nu)
            dKe_dt = (Kp - Km) / (2.0 * dt)

            area        = tri_area(c1, c2, c3)
            dm_node_dt  = rho * area / 3.0
            L_char      = np.sqrt(4.0 * area / np.sqrt(3.0))
            drot_dt     = dm_node_dt * (L_char ** 2) / 12.0

            n_n = 3
        else:  # QUAD4 / QUAD
            c1, c2, c3, c4 = crds
            Kp = _element_K_mitc4_plus(c1, c2, c3, c4, t + dt, E, nu)
            Km = _element_K_mitc4_plus(c1, c2, c3, c4, t - dt, E, nu)
            dKe_dt = (Kp - Km) / (2.0 * dt)

            a1 = 0.5 * np.linalg.norm(np.cross(crds[1]-crds[0], crds[2]-crds[0]))
            a2 = 0.5 * np.linalg.norm(np.cross(crds[2]-crds[0], crds[3]-crds[0]))
            area        = a1 + a2
            dm_node_dt  = rho * area / 4.0
            L_char      = np.sqrt(area)
            drot_dt     = dm_node_dt * (L_char ** 2) / 12.0

            n_n = 4

        dMe_dt = np.zeros(n_n * 6)
        for i in range(n_n):
            dMe_dt[i*6 : i*6+3] = dm_node_dt   # translational
            dMe_dt[i*6+3 : i*6+6] = drot_dt    # rotational

        return dKe_dt, dMe_dt

    # ------------------------------------------------------------------
    # Core eigensensitivity scatter
    # ------------------------------------------------------------------

    def _scatter_eigensens(
        self,
        nids: List[int],
        dKe_dt: np.ndarray,
        dMe_dt: np.ndarray,
    ) -> np.ndarray:
        """
        ∂λᵢ/∂tₑ = φᵢ[free]ᵀ dKe_free φᵢ[free]  −  λᵢ · Σ dMe_free · φᵢ²

        Scatters element-level (18×18 or 24×24) dKe into free-DOF subspace.

        Returns (n_modes,).
        """
        gmap        = self.global_to_free
        nid_to_idx  = self.nid_to_idx

        # Global DOF list for all element nodes (local index → global DOF)
        global_dofs = np.array(
            [nid_to_idx[nid] * 6 + d for nid in nids for d in range(6)]
        )

        # Identify which element-local DOFs are free
        pairs = [(loc, gmap[int(g)])
                 for loc, g in enumerate(global_dofs)
                 if int(g) in gmap]
        if not pairs:
            return np.zeros(self.n_modes)

        local_idx, free_idx = zip(*pairs)
        local_idx = list(local_idx)
        free_idx  = list(free_idx)

        # Sub-block of element stiffness/mass derivative in free-DOF space
        dKe_free = dKe_dt[np.ix_(local_idx, local_idx)]   # (nf, nf)
        dMe_free = dMe_dt[local_idx]                        # (nf,)

        result = np.zeros(self.n_modes)
        for i in range(self.n_modes):
            phi      = self.vecs_free[free_idx, i]          # (nf,)
            dK_term  = phi @ dKe_free @ phi
            dM_term  = np.dot(phi ** 2, dMe_free)
            result[i] = dK_term - self.lambdas[i] * dM_term

        return result
