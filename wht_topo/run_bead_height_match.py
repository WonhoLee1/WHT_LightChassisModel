# -*- coding: utf-8 -*-
"""
run_bead_height_match.py — 비드 영역 높이 조정으로 목표 모드 매칭
====================================================================

이미 비드가 형성된 메쉬에서, 지정한 노드셋(비드 영역) 노드들의 Z-offset(높이)만
연속 design variable로 두고 나머지 노드는 고정한 채, JAX 자동미분
(wht_solver.wht_eigensolver.make_modal_freq_fn)으로 목표 차수별 고유진동수에
수렴시킨다.

두 가지 모드:
  (a) --target-model 지정: 별도 FEM 결과에서 목표 주파수 + 목표 모드 형상을 모두
      추출 → MAC 소프트 어사인 기반 wht_solver.WHTOptimizer 사용 (모드 교차에 안전).
  (b) --target-freqs 만 지정: 목표 형상 없이 차수 순서를 그대로 가정한 단순 MSE
      손실로 직접 최적화 (빠름, 모드 교차가 없다고 가정할 수 있을 때 사용).

실행 예시
---------
  python wht_topo/run_bead_height_match.py --model chassis.k --bead-node-set 100 \
      --target-freqs 45.0 80.0 120.0 --n-steps 200

  python wht_topo/run_bead_height_match.py --model chassis.k --bead-node-set 100 \
      --target-model target_chassis.k --num-modes 3 --n-steps 200
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="비드가 형성된 LS-DYNA 메쉬 (.k)")
    p.add_argument("--bead-node-set", type=int, required=True,
                    help="높이를 조정할 비드 영역 노드셋 ID (*SET_NODE)")
    p.add_argument("--target-freqs", type=float, nargs="+", default=None,
                    help="차수별 목표 고유진동수 [Hz] (예: 1차 2차 3차 순서)")
    p.add_argument("--target-model", default=None,
                    help="목표 모드 형상까지 사용할 별도 FEM 결과 (.k, 동일 메쉬 가정)")
    p.add_argument("--num-modes", type=int, default=None,
                    help="맞출 모드 차수 개수 (기본: target-freqs 길이 또는 3)")
    p.add_argument("--z-min", type=float, default=-20.0, help="비드 높이 하한 [mm]")
    p.add_argument("--z-max", type=float, default=20.0, help="비드 높이 상한 [mm]")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--monitor", action="store_true", help="PyVistaQt 실시간 모니터 사용")
    return p.parse_args()


def load_model(path: str):
    from wht_modeler.io import LSDYNAReader
    return LSDYNAReader().read(path)


def build_free_node_mask(model, bead_node_set: int) -> np.ndarray:
    sorted_nids = sorted(model.nodes.keys())
    bead_nids = set(model.get_nodes_by_set(bead_node_set))
    if not bead_nids:
        raise ValueError(f"노드셋 {bead_node_set}이 비어있거나 존재하지 않습니다.")
    return np.array([nid in bead_nids for nid in sorted_nids], dtype=bool)


def run_freq_only(model, free_mask: np.ndarray, target_freqs: list, args) -> None:
    """목표 모드 형상 없이, 차수 순서를 그대로 가정한 단순 MSE 최적화."""
    from wht_solver.wht_eigensolver import make_modal_freq_fn
    from wht_solver.wht_optimizer import DesignVariables, DesignBounds, clip_to_bounds
    import optax

    num_modes = args.num_modes or len(target_freqs)
    sorted_eids = sorted(model.elements.keys())
    sorted_nids = sorted(model.nodes.keys())
    t0 = np.array([model.properties[model.elements[eid].pid].t for eid in sorted_eids])
    mat = next(iter(model.materials.values()))

    freq_fn = make_modal_freq_fn(model, num_modes=num_modes)
    bounds = DesignBounds(
        t_min=float(t0.min()), t_max=float(t0.max()),
        z_min=args.z_min, z_max=args.z_max,
        E_min=mat.E, E_max=mat.E,
        rho_min=mat.rho, rho_max=mat.rho,
        free_node_mask=free_mask,
    )
    free_mask_j = jnp.asarray(free_mask)
    dv = DesignVariables(
        t_field=jnp.array(t0), z_offsets=jnp.zeros(len(sorted_nids)),
        E=mat.E, rho=mat.rho,
    )
    target = jnp.array(target_freqs[:num_modes])

    def loss_fn(d: DesignVariables) -> jnp.ndarray:
        freqs_opt = freq_fn(d.t_field, d.z_offsets, jnp.array(d.E), jnp.array(d.rho))
        return jnp.mean(jnp.square(freqs_opt - target))

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(dv)
    loss_and_grad = jax.value_and_grad(loss_fn)

    print(f"[freq-only] {args.n_steps} steps, target={target_freqs[:num_modes]} Hz")
    for step in range(1, args.n_steps + 1):
        loss_val, grads = loss_and_grad(dv)
        grads = DesignVariables(
            t_field=grads.t_field, z_offsets=grads.z_offsets * free_mask_j,
            E=grads.E, rho=grads.rho,
        )
        updates, opt_state = optimizer.update(grads, opt_state, dv)
        dv = optax.apply_updates(dv, updates)
        dv = clip_to_bounds(dv, bounds)
        if step % args.log_every == 0 or step == 1:
            freqs_now = freq_fn(dv.t_field, dv.z_offsets, jnp.array(dv.E), jnp.array(dv.rho))
            print(f"  step {step:4d}/{args.n_steps}  loss={float(loss_val):.6e}  "
                  f"freqs={[round(float(f), 2) for f in freqs_now]}")

    freqs_final = freq_fn(dv.t_field, dv.z_offsets, jnp.array(dv.E), jnp.array(dv.rho))
    print(f"최종 주파수: {[round(float(f), 3) for f in freqs_final]} Hz "
          f"(목표: {target_freqs[:num_modes]} Hz)")


def run_with_target_shapes(model, free_mask: np.ndarray, target_model_path: str, args) -> None:
    """별도 FEM 결과의 모드 형상까지 사용하는 MAC 소프트 어사인 최적화."""
    from wht_solver.wht_solver import WHTSolver
    from wht_solver.wht_optimizer import (
        DesignVariables, DesignBounds, WHTOptimizer,
    )
    from wht_solver.wht_mapper import WHTMapper
    from wht_solver.wht_monitor import OptimizationMonitor

    num_modes = args.num_modes or 3
    target_model = load_model(target_model_path)
    target_result = WHTSolver(target_model).solve_modal(num_modes=num_modes)

    sorted_eids = sorted(model.elements.keys())
    sorted_nids = sorted(model.nodes.keys())
    t0 = np.array([model.properties[model.elements[eid].pid].t for eid in sorted_eids])
    mat = next(iter(model.materials.values()))

    bounds = DesignBounds(
        t_min=float(t0.min()), t_max=float(t0.max()),
        z_min=args.z_min, z_max=args.z_max,
        E_min=mat.E, E_max=mat.E,
        rho_min=mat.rho, rho_max=mat.rho,
        free_node_mask=free_mask,
    )
    monitor = None
    if args.monitor:
        monitor = OptimizationMonitor()
        monitor.init_mesh(model)

    # target_model이 base 메쉬와 노드 수/순서가 달라도 동작하도록, target 노드 좌표를
    # 함께 넘겨 WHTOptimizer가 RBF로 base 메쉬 노드 위치에 모드 형상을 매핑하게 한다
    # (동일 메쉬라도 무해 — 좌표가 같으면 보간 결과가 원래 값과 거의 일치).
    optimizer = WHTOptimizer(
        base_model=model,
        target_results={"modal": target_result},
        mapper=WHTMapper(),
        bounds=bounds,
        load_cases=[],
        num_modes=num_modes,
        lr=args.lr,
        weights={"freq": 1.0, "mac": 0.0, "static": 0.0, "smooth": 0.0},
        monitor=monitor,
        target_node_coords=target_model.nodes_array(),
    )

    init_vars = DesignVariables(
        t_field=jnp.array(t0), z_offsets=jnp.zeros(len(sorted_nids)),
        E=mat.E, rho=mat.rho,
    )
    final_vars, loss_history = optimizer.run(
        init_vars, n_steps=args.n_steps, log_every=args.log_every,
    )

    final_result = WHTSolver(model).solve_modal(num_modes=num_modes)
    print(f"최종 주파수: {final_result.frequencies[:num_modes]} Hz "
          f"(목표: {target_result.frequencies[:num_modes]} Hz)")
    print(f"최종 loss: {loss_history[-1]:.6e}")


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    free_mask = build_free_node_mask(model, args.bead_node_set)
    print(f"비드 영역 자유노드: {int(free_mask.sum())} / {len(free_mask)}")

    if args.target_model:
        run_with_target_shapes(model, free_mask, args.target_model, args)
    elif args.target_freqs:
        run_freq_only(model, free_mask, args.target_freqs, args)
    else:
        print("오류: --target-freqs 또는 --target-model 중 하나는 필요합니다.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
