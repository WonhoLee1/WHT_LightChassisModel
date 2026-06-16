# -*- coding: utf-8 -*-
"""run_bead_height_match.py 핵심 경로 스모크 테스트 (합성 5x5 평판 메쉬).

solve_modal() 내부에서 multiprocessing.Process를 사용하므로 Windows(spawn)에서는
반드시 __main__ 가드 안에서 실행해야 한다.
"""
import argparse
import numpy as np


def main():
    from wht_modeler.wht_mesh_model import WHTMeshModel
    from wht_topo.run_bead_height_match import run_freq_only

    NX, NY = 6, 6
    DX = 50.0

    model = WHTMeshModel()
    nid = 1
    node_grid = {}
    for j in range(NY):
        for i in range(NX):
            model.add_node(nid, i * DX, j * DX, 0.0)
            node_grid[(i, j)] = nid
            nid += 1

    model.add_material(1, E=210000.0, nu=0.3, rho=7.8e-9)
    model.add_property(1, "SHELL", t=1.0, mid=1)

    eid = 1
    for j in range(NY - 1):
        for i in range(NX - 1):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            model.add_element(eid, [n1, n2, n3, n4], elem_type="QUAD4", pid=1)
            eid += 1

    # Fix the four corners fully (cantilevered plate, free elsewhere)
    corners = [node_grid[(0, 0)], node_grid[(NX - 1, 0)],
               node_grid[(0, NY - 1)], node_grid[(NX - 1, NY - 1)]]
    model.apply_spc(corners, dofs=(0, 1, 2, 3, 4, 5))

    # "Bead region" = interior nodes only
    interior = [node_grid[(i, j)] for i in range(1, NX - 1) for j in range(1, NY - 1)]
    model.add_node_set_by_name("bead_region", interior)
    interior_set = set(interior)
    free_mask = np.array([nid in interior_set for nid in sorted(model.nodes.keys())], dtype=bool)
    print(f"free nodes: {free_mask.sum()} / {len(free_mask)}")

    args = argparse.Namespace(
        num_modes=2, z_min=-5.0, z_max=5.0, n_steps=8, lr=0.02, log_every=2,
    )
    run_freq_only(model, free_mask, [60.0, 90.0], args)
    print("SMOKE TEST DONE")


if __name__ == "__main__":
    main()
