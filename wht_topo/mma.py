# -*- coding: utf-8 -*-
"""
mma.py
======
MMA (Method of Moving Asymptotes) 기반 최적화 업데이트 엔진.

OC-like Simplified MMA 변형으로, 위상 최적화에 특화된 구현입니다.
비정형 메시의 요소 면적 가중치를 고려한 정밀 체적 제약을 지원합니다.
"""

import jax.numpy as jnp


class MMAOptimizer:
    """
    Simplified MMA 업데이트 엔진.

    Parameters
    ----------
    n_vars : int
        설계 변수(요소) 수.
    n_constraints : int
        제약 조건 수 (기본: 1 = 체적 제약).
    vol_frac : float
        목표 체적 분율. MMA 내부 이분법 탐색의 기준값.
    move : float
        설계 변수 이동 한계 (0~1).
    """

    def __init__(self, n_vars: int, n_constraints: int = 1,
                 vol_frac: float = 0.3, move: float = 0.2):
        self.n = n_vars
        self.m = n_constraints
        self.vol_frac = vol_frac
        self.move = move

        # Asymptotes
        self.low = jnp.zeros(n_vars)
        self.upp = jnp.ones(n_vars)
        self.xold1 = jnp.zeros(n_vars)
        self.xold2 = jnp.zeros(n_vars)

        # 변수 하한/상한
        self.alb = jnp.full(n_vars, 1e-3)  # 밀도 하한 (0 방지)
        self.aub = jnp.ones(n_vars)

    def update(self, x: jnp.ndarray, f0val: float, df0dx: jnp.ndarray,
               fval: jnp.ndarray, dfdx: jnp.ndarray, iteration: int,
               project_fn=None) -> jnp.ndarray:
        """
        MMA 업데이트 1회 수행.

        Parameters
        ----------
        x : (N,) 현재 밀도
        f0val : float  목적 함수 값
        df0dx : (N,) 목적 함수 구배
        fval : (M,) 제약 함수 값 (음수이면 만족, 현재 미사용)
        dfdx : (N,) 체적 제약 구배 (= 1/N)
        iteration : int  현재 반복 번호 (1-indexed)
        project_fn : callable(x) → x_proj, optional
            이산 투사 함수. 지정 시 이분법에서 mean(project(x_test))로
            체적을 계산하여 시각적 비드 면적 = vol_frac 을 정확히 강제.
            None이면 기존 방식 mean(x_test) 사용.

        Returns
        -------
        x_new : (N,) 업데이트된 밀도
        """
        import numpy as np

        # ── 1. Asymptotes 업데이트 ──
        if iteration <= 2:
            self.low = x - self.move
            self.upp = x + self.move
        else:
            # 진동 감지: 부호가 같으면 가속, 다르면 감속
            zz = (x - self.xold1) * (self.xold1 - self.xold2)
            factor_low = jnp.where(zz > 0, 1.2, 0.7)
            factor_upp = jnp.where(zz > 0, 1.2, 0.7)
            self.low = x - factor_low * (self.xold1 - self.low)
            self.upp = x + factor_upp * (self.upp - self.xold1)
            # 안전 클리핑
            self.low = jnp.clip(self.low, x - 10.0, x - 0.01)
            self.upp = jnp.clip(self.upp, x + 0.01, x + 10.0)

        # ── 2. 이분법 기반 Lagrange Multiplier 탐색 ──
        safe_df0dx = jnp.minimum(df0dx, -1e-10)

        alpha = jnp.maximum(self.alb, jnp.maximum(self.low + 0.1 * (x - self.low), x - self.move))
        beta_ = jnp.minimum(self.aub, jnp.minimum(self.upp - 0.1 * (self.upp - x), x + self.move))

        # 체적 계산 함수: project_fn 제공 시 x_proj 기준, 아니면 mean(x) 기준
        # dfdx에 projection gradient가 포함될 수 있으므로 sum(x*dfdx) 대신 mean(x) 사용
        def _vol(x_t):
            if project_fn is not None:
                return float(jnp.mean(project_fn(x_t)))
            return float(jnp.mean(x_t))

        l1, l2 = 1e-10, 1e12
        for _ in range(60):
            l_mid = np.sqrt(l1 * l2)  # geometric mean: uniform convergence across scale
            step = jnp.sqrt(-safe_df0dx / (l_mid * dfdx + 1e-10))
            x_test = jnp.clip(x * step, alpha, beta_)
            if _vol(x_test) > self.vol_frac:
                l1 = l_mid
            else:
                l2 = l_mid

        step_final = jnp.sqrt(-safe_df0dx / (l2 * dfdx + 1e-10))
        x_new = jnp.clip(x * step_final, alpha, beta_)

        # ── 3. 이력 업데이트 ──
        self.xold2 = self.xold1
        self.xold1 = x

        return x_new


if __name__ == "__main__":
    pass
