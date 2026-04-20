# -*- coding: utf-8 -*-
"""
================================================================================
SolverVerification / analytical_solutions.py
================================================================================
■ 목적
    쉘(판) 이론의 해석적(Analytical) 해를 제공합니다.
    단위계: mm / MPa(N/mm²) / tonne / N
================================================================================
"""
import numpy as np

def kirchhoff_plate_field_solution(
    Lx: float, Ly: float, q: float,
    E: float, t: float, nu: float,
    x_coords: np.ndarray, y_coords: np.ndarray,
    n_terms: int = 15
) -> dict:
    """단순지지 사각형 판의 모든 노드 위치에서 이론적 해(필드)를 계산합니다."""
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    w = np.zeros_like(x_coords)
    kx = np.zeros_like(x_coords)
    ky = np.zeros_like(x_coords)
    kxy = np.zeros_like(x_coords)

    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            q_mn = 16.0 * q / (np.pi**2 * m * n)
            denom = np.pi**4 * D * ((m / Lx)**2 + (n / Ly)**2)**2
            coeff = q_mn / denom
            
            smx = np.sin(m * np.pi * x_coords / Lx)
            sny = np.sin(n * np.pi * y_coords / Ly)
            cmx = np.cos(m * np.pi * x_coords / Lx)
            cny = np.cos(n * np.pi * y_coords / Ly)
            
            w += coeff * smx * sny
            kx += coeff * (m * np.pi / Lx)**2 * smx * sny
            ky += coeff * (n * np.pi / Ly)**2 * smx * sny
            kxy -= coeff * (m * np.pi / Lx) * (n * np.pi / Ly) * cmx * cny

    factor_eps = t/2.0
    ex = kx * factor_eps
    ey = ky * factor_eps
    exy = kxy * factor_eps
    
    factor_sig = E / (1.0 - nu**2)
    sx = factor_sig * (ex + nu * ey)
    sy = factor_sig * (ey + nu * ex)
    sxy = (E / (2.0 * (1.0 + nu))) * exy
    vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3.0*sxy**2)

    return {
        'w': w, 'strain_x': ex, 'stress_x': sx, 'stress_y': sy, 'stress_xy': sxy, 'stress_vm': vm
    }

def kirchhoff_frequency(Lx, Ly, E, t, nu, rho, m=1, n=1):
    """단순지지 사각형 판의 고유 진동수 (Hz)."""
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    # w = pi^2 * sqrt(D/rho*t) * [(m/Lx)^2 + (n/Ly)^2]
    val = (np.pi**2) * np.sqrt(D / (rho * t)) * ((m / Lx)**2 + (n / Ly)**2)
    # f = w / (2*pi) = (pi/2) * sqrt(D/rho*t) * [...]
    return val / (2.0 * np.pi)

def beam_3point_bending_deflection(L, E, I, P):
    return (P * L**3) / (48.0 * E * I)

def beam_3point_bending_stress(L, b, t, P):
    return (3.0 * P * L) / (2.0 * b * t**2)

def beam_twisting_deflection(F, Lx, Ly, E, t, nu):
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    return (F * Lx * Ly) / (2.0 * D * (1.0 - nu))
