# -*- coding: utf-8 -*-
"""target_node_coords 기반 cross-mesh MAC 매핑 스모크 테스트.

base 메쉬(6x6, 50mm 간격)와 target 메쉬(9x9, 40mm 간격, 노드 수도 좌표도 다름)로
WHTOptimizer가 RBF 매핑을 거쳐 phis_target을 base 노드 공간으로 정렬하는지 확인.
"""
import numpy as np


def build_plate(nx, ny, dx):
    from wht_modeler.wht_mesh_model import WHTMeshModel
    model = WHTMeshModel()
    nid = 1
    grid = {}
    rng = np.random.RandomState(0)
    for j in range(ny):
        for i in range(nx):
            # Small z variation: flat plate (z=0 everywhere) makes the RBF's
            # affine polynomial term rank-deficient (no z variation to fit).
            z = 0.0
            model.add_node(nid, i * dx, j * dx, z)
            grid[(i, j)] = nid
            nid += 1
    model.add_material(1, E=210000.0, nu=0.3, rho=7.8e-9)
    model.add_property(1, "SHELL", t=1.0, mid=1)
    eid = 1
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1, n2, n3, n4 = grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]
            model.add_element(eid, [n1, n2, n3, n4], elem_type="QUAD4", pid=1)
            eid += 1
    corners = [grid[(0, 0)], grid[(nx - 1, 0)], grid[(0, ny - 1)], grid[(nx - 1, ny - 1)]]
    model.apply_spc(corners, dofs=(0, 1, 2, 3, 4, 5))
    return model, grid


def main():
    from wht_solver.wht_solver import WHTSolver
    from wht_solver.wht_optimizer import DesignVariables, DesignBounds, WHTOptimizer
    from wht_solver.wht_mapper import WHTMapper

    base_model, base_grid = build_plate(6, 6, 50.0)
    target_model, _ = build_plate(9, 9, 40.0)   # different node count AND physical size

    print(f"base nodes={len(base_model.nodes)}  target nodes={len(target_model.nodes)}")

    num_modes = 2
    target_result = WHTSolver(target_model).solve_modal(num_modes=num_modes)
    print(f"target freqs: {target_result.frequencies[:num_modes]}")

    interior = [base_grid[(i, j)] for i in range(1, 5) for j in range(1, 5)]
    interior_set = set(interior)
    sorted_nids = sorted(base_model.nodes.keys())
    free_mask = np.array([n in interior_set for n in sorted_nids], dtype=bool)

    sorted_eids = sorted(base_model.elements.keys())
    t0 = np.array([base_model.properties[base_model.elements[e].pid].t for e in sorted_eids])
    mat = next(iter(base_model.materials.values()))

    bounds = DesignBounds(
        t_min=float(t0.min()), t_max=float(t0.max()),
        z_min=-5.0, z_max=5.0,
        E_min=mat.E, E_max=mat.E, rho_min=mat.rho, rho_max=mat.rho,
        free_node_mask=free_mask,
    )

    optimizer = WHTOptimizer(
        base_model=base_model,
        target_results={"modal": target_result},
        mapper=WHTMapper(),
        bounds=bounds,
        load_cases=[],
        num_modes=num_modes,
        lr=0.02,
        weights={"freq": 1.0, "mac": 0.0, "static": 0.0, "smooth": 0.0},
        target_node_coords=target_model.nodes_array(),
    )

    init_vars = DesignVariables(
        t_field=np.array(t0), z_offsets=np.zeros(len(sorted_nids)),
        E=mat.E, rho=mat.rho,
    )
    final_vars, loss_history = optimizer.run(init_vars, n_steps=4, log_every=1)
    print("loss history:", loss_history)
    print("CROSS-MESH MAC SMOKE TEST DONE")


if __name__ == "__main__":
    main()
