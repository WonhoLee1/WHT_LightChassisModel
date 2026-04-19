# -*- coding: utf-8 -*-
"""TRIA3 element K ?? ??? (??? ?? ??)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_solver.wht_tria3_element import (
    _element_K_tria3,
    K_tria3_scipy,
    M_tria3_lumped,
)

# ---------- ?? stub ----------
class Node:
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
class Elem:
    def __init__(self, t, nids, pid=1): self.type = t; self.node_ids = nids; self.pid = pid
class Prop:
    def __init__(self, t, mid): self.t = t; self.mid = mid
class Mat:
    def __init__(self, E, nu, rho=7.85e-9): self.E=E; self.nu=nu; self.rho=rho
class Model:
    def __init__(self, coords, elems, E=210000., nu=0.3, t=1.):
        self.nodes      = {i+1: Node(*c) for i,c in enumerate(coords)}
        self.elements   = {i+1: e for i,e in enumerate(elems)}
        self.properties = {1: Prop(t=t, mid=1)}
        self.materials  = {1: Mat(E=E, nu=nu)}

def check(label, cond, detail=''):
    tag = 'OK' if cond else 'FAIL'
    print(f"  [{tag}] {label}" + (f" - {detail}" if detail else ''))
    return cond

# ==========================================================================
# Test 1: K ??? + ?? ??(6) + ????
# ==========================================================================
def test_K_properties():
    print("\n--- Test 1: K matrix properties (symmetry, rank, PSD) ---")
    c1 = np.array([0.,0.,0.]); c2 = np.array([2.,0.,0.]); c3 = np.array([1.,2.,0.])
    K = _element_K_tria3(c1, c2, c3, t=1., E=210000., nu=0.3)

    ok = True
    ok &= check("K is 18x18",        K.shape == (18,18))
    ok &= check("K is symmetric",    np.allclose(K, K.T, atol=1e-8))

    # ?? ?? 6?: rank = 18-6 = 12
    rank = np.linalg.matrix_rank(K, tol=1e-4)
    ok &= check("rank(K) == 12",     rank == 12, f"got {rank}")

    # ?? ??? >= 0 (????)
    eigs = np.linalg.eigvalsh(K)
    ok &= check("K is PSD (min eig >= -1e-6)", eigs.min() >= -1e-6, f"min={eigs.min():.3e}")
    return ok

# ==========================================================================
# Test 2: ?? ???? ? K*u = 0
# ==========================================================================
def test_rigid_body():
    print("\n--- Test 2: Rigid body translation -> K*u = 0 ---")
    c1 = np.array([0.,0.,0.]); c2 = np.array([2.,0.,0.]); c3 = np.array([1.,2.,0.])
    K = _element_K_tria3(c1, c2, c3, t=1., E=210000., nu=0.3)

    ok = True
    for axis, label in [(0,'TX'),(1,'TY'),(2,'TZ')]:
        u = np.zeros(18)
        for n in range(3): u[n*6+axis] = 1.0   # pure translation
        residual = np.linalg.norm(K @ u)
        ok &= check(f"K*u_{label} ~ 0", residual < 1e-6, f"|r|={residual:.2e}")
    return ok

# ==========================================================================
# Test 3: ?? ?? ? ?? ???? ??
# ==========================================================================
def test_plane_tension():
    print("\n--- Test 3: K_tria3_scipy ? in-plane tension ---")
    E, nu, t, e0 = 210000., 0.3, 1., 0.001
    coords = [[0,0,0],[2,0,0],[0,2,0]]
    elems  = [Elem("TRIA3",[1,2,3])]
    model  = Model(coords, elems, E=E, nu=nu, t=t)
    sorted_nids = [1,2,3]
    nid_to_idx  = {1:0, 2:1, 3:2}

    K = K_tria3_scipy(model, sorted_nids, nid_to_idx).toarray()
    ok = check("K assembled (18x18)", K.shape == (18,18))

    # u_x = e0*x: ??1=(0), ??2=(2*e0), ??3=(0)
    u = np.zeros(18)
    u[6] = 2.*e0          # node2 u_x (DOF index = 1*6+0)
    f = K @ u
    ok &= check("K*u finite", np.all(np.isfinite(f)))
    return ok

# ==========================================================================
# Test 4: ?? ?? ? ?? ?? ??
# ==========================================================================
def test_lumped_mass():
    print("\n--- Test 4: M_tria3_lumped ? total mass check ---")
    E, nu, t, rho = 210000., 0.3, 2., 7.85e-6
    coords = [[0,0,0],[2,0,0],[0,2,0]]
    elems  = [Elem("TRIA3",[1,2,3])]
    model  = Model(coords, elems, E=E, nu=nu, t=t)
    model.materials[1].rho = rho
    sorted_nids = [1,2,3]
    nid_to_idx  = {1:0, 2:1, 3:2}

    ndof = 3*6
    M = M_tria3_lumped(model, ndof, sorted_nids, nid_to_idx)
    area = 0.5*2.*2.
    expected_mass = area * t * rho
    ok = check("M vector length == ndof", len(M) == ndof)
    # ? ??? ??? total_mass? ??? ?
    ok &= check("Total X-mass correct",
                abs(M[0::6].sum() - expected_mass) < 1e-10,
                f"got={M[0::6].sum():.4e}, exp={expected_mass:.4e}")
    return ok

# ==========================================================================
# Test 5: ?? ?? (QUAD4 + TRIA3) ? K_tria3_scipy ? QUAD4 ???
# ==========================================================================
def test_mixed_mesh():
    print("\n--- Test 5: Mixed QUAD4+TRIA3 ? only TRIA3 assembled ---")
    coords = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0]]
    elems  = [Elem("QUAD4",[1,2,3,4]), Elem("TRIA3",[2,5,3])]
    model  = Model(coords, elems)
    sorted_nids = [1,2,3,4,5]
    nid_to_idx  = {n:i for i,n in enumerate(sorted_nids)}

    K = K_tria3_scipy(model, sorted_nids, nid_to_idx)
    ndof = 5*6
    ok = check("shape correct", K.shape == (ndof, ndof))

    # QUAD4 ??(1,4)? ? TRIA3? ?? ? ? ? ?? DOF ?? 0
    row_node1 = K[0:6, :].toarray()
    ok &= check("Node 1 (QUAD-only) rows are zero", np.allclose(row_node1, 0))
    return ok

# ==========================================================================
if __name__ == "__main__":
    results = [
        test_K_properties(),
        test_rigid_body(),
        test_plane_tension(),
        test_lumped_mass(),
        test_mixed_mesh(),
    ]
    n = sum(results)
    print(f"\n{'='*50}")
    print(f"TRIA3 element tests: {n}/{len(results)} passed")
