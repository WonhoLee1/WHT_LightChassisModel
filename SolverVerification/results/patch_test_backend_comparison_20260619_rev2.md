# WHT Solver Patch Test — Backend Comparison Report (Rev.2)

**Date:** 2026-06-19 (Rev.2 — Numba P_proj bugfix applied)  
**Solver:** WHT QUAD4 MITC4+ Flat Shell (+ TRIA3 CST)  
**Element formulation:** MITC4+ with Rigid Body Projection (P_proj), MITC shear locking prevention, Selective Reduced Integration (SRI)  
**Backends tested:** JAX (vmap/jit), NumPy (pure), Numba (nopython JIT)

---

## Summary

| Backend | Pass / Total | Max diff vs JAX |
|---------|-------------|-----------------|
| JAX     | **22 / 22** | — (reference)   |
| NumPy   | **22 / 22** | 0.0000%         |
| Numba   | **22 / 22** | 0.0000%         |

모든 백엔드가 동일한 수치 결과를 산출하며 22개 패치 테스트를 전부 통과하였다.

---

## Bug Fixes Applied

### Fix 1 (Rev.1, 2026-06-19) — NumPy `gp` 변수 미정의

`_element_K_mitc4_plus_np`에서 Gauss 적분점 배열이 주석 처리되어 있었음.

```python
# 수정 전 (NameError)
# gp = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
# 수정 후
gp = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
```

### Fix 2 (Rev.1, 2026-06-19) — NumPy / Numba P_proj 누락

JAX에만 구현되어 있던 강체 투영 행렬 `P_proj = I - Q@Q^T`를 NumPy·Numba 버전에 추가.

### Fix 3 (Rev.2, 2026-06-19) — **Numba R_rbm 3D 좌표 누락 (핵심 버그)**

**원인:** Numba `_element_K_mitc4_plus_nb`의 R_rbm 구성 시  
- JAX/NumPy: `coords_loc = (coords - coords[0]) @ T_mat.T` (3D local, **pz 포함**)  
- Numba (버그): `coords_2d` (2D 투영, **pz = 0 강제**)

Flat mesh(패치테스트)에서는 pz ≈ 0이므로 차이 없어 탐지 불가. 실제 비드 패널처럼 warped element가 있는 3D 메시에서만 오차 발생.

**영향:** ccx_iter016 Mode 7 기준  
- 수정 전 Numba: **6.9980 Hz** (+42% vs CCX 4.9344 Hz)  
- 수정 후 Numba: **4.9376 Hz** (+0.07% vs CCX) — JAX와 완전 일치

**수정 코드** (`wht_quad4_element.py`, `_element_K_mitc4_plus_nb`):

```python
# 수정 전 (pz=0)
R_rbm = np.zeros((24, 6))
for i in range(4):
    px = coords_2d[i, 0]
    py = coords_2d[i, 1]
    R_rbm[6*i+0, 5] = -py
    R_rbm[6*i+1, 5] = px
    R_rbm[6*i+2, 3] = py;  R_rbm[6*i+2, 4] = -px
    ...

# 수정 후 (3D local coords, pz 포함)
coords_pz = np.zeros(4)
coords_pz[1] = p2[0]*z_loc[0] + p2[1]*z_loc[1] + p2[2]*z_loc[2]
coords_pz[2] = p3[0]*z_loc[0] + p3[1]*z_loc[1] + p3[2]*z_loc[2]
coords_pz[3] = p4[0]*z_loc[0] + p4[1]*z_loc[1] + p4[2]*z_loc[2]
R_rbm = np.zeros((24, 6))
for i in range(4):
    px = coords_2d[i, 0];  py = coords_2d[i, 1];  pz = coords_pz[i]
    R_rbm[6*i+0, 4] = pz;  R_rbm[6*i+0, 5] = -py
    R_rbm[6*i+1, 3] = -pz; R_rbm[6*i+1, 5] = px
    R_rbm[6*i+2, 3] = py;  R_rbm[6*i+2, 4] = -px
    ...
```

---

## Patch Test Results — QUAD4

### JAX / NumPy / Numba (identical results — max diff 0.0000%)

| Test | Quantity | Theory | FEM | Error |
|------|----------|--------|-----|-------|
| 3-pt Bending | Max Deflection (mm) | 1.488 | 1.486 | 0.17% |
| 3-pt Bending | Max Stress Sx (MPa) | 375 | 358.2 | 4.48% |
| Membrane Tension | Max Displacement X (mm) | 0.04762 | 0.04762 | 0.00% |
| Membrane Tension | Avg Stress Sx (MPa) | 100 | 100 | 0.00% |
| 4-pt Bending | Max Deflection (mm) | 10.14 | 9.924 | 2.14% |
| Plate Twisting | Corner Deflection (mm) | 37.14 | 37.19 | 0.14% |
| Natural Frequency | Mode 1 (1,1) [Hz] | 49.17 | 49.04 | 0.27% |
| Natural Frequency | Mode 2 (1,2) [Hz] | 122.9 | 122.7 | 0.18% |
| Natural Frequency | Mode 3 (2,1) [Hz] | 122.9 | 122.7 | 0.18% |
| Natural Frequency | Mode 4 (2,2) [Hz] | 196.7 | 195.2 | 0.78% |
| Natural Frequency | Mode 5 (1,3) [Hz] | 245.9 | 246.4 | 0.22% |

**Result: 11/11 PASS**

---

## Patch Test Results — TRIA3

### JAX / NumPy / Numba (identical results — max diff 0.0000%)

| Test | Quantity | Theory | FEM | Error |
|------|----------|--------|-----|-------|
| 3-pt Bending | Max Deflection (mm) | 1.488 | 1.486 | 0.13% |
| 3-pt Bending | Max Stress Sx (MPa) | 375 | 358.4 | 4.41% |
| Membrane Tension | Max Displacement X (mm) | 0.04762 | 0.04814 | 1.10% |
| Membrane Tension | Avg Stress Sx (MPa) | 100 | 100.2 | 0.24% |
| 4-pt Bending | Max Deflection (mm) | 10.14 | 9.924 | 2.14% |
| Plate Twisting | Corner Deflection (mm) | 37.14 | 37.96 | 2.21% |
| Natural Frequency | Mode 1 (1,1) [Hz] | 49.17 | 49.28 | 0.22% |
| Natural Frequency | Mode 2 (1,2) [Hz] | 122.9 | 123.9 | 0.77% |
| Natural Frequency | Mode 3 (2,1) [Hz] | 122.9 | 123.9 | 0.77% |
| Natural Frequency | Mode 4 (2,2) [Hz] | 196.7 | 200.1 | 1.72% |
| Natural Frequency | Mode 5 (1,3) [Hz] | 245.9 | 249.1 | 1.33% |

**Result: 11/11 PASS**

---

## Accuracy Notes

- **굽힘 응력 오차 ~4.4%**: QUAD4 4×4 메시 기준. Saint-Venant 응력 집중 및 메시 밀도 효과. 허용 기준(10%) 이내.
- **TRIA3 Mode 4 오차 1.72%**: 상수 변형률 삼각형의 대칭 모드 과강성 경향. 품질 기준(3%) 이내.
- **백엔드 간 편차 0.0000%**: 수치 일치 확인 (~4.4×10⁻¹⁶, 기계 엡실론 수준).

---

## Modal Frequencies — Full Spectrum (QUAD4, first 10 modes)

```
[  0.451   49.040  122.706  122.706  195.150  246.386  246.387  316.871  316.871  421.445 ] Hz
```

Mode 1 (~0.45 Hz): rigid body residual (spring-free boundary, numerical artifact)  
Modes 2–10: flexural/membrane plate modes

---

## Benchmark — Small Shell (50×50 QUAD4)

**Model:** 50×50 QUAD4 simply-supported square plate (1000×1000mm, t=5mm)  
**Size:** 2,601 nodes / 2,500 elements / ~15,606 DOF  
**Solver:** ARPACK, 10 modes

| Backend | K Assembly [ms] | Modal Solve [ms] | Total [ms] | vs Numba |
|---------|-----------------|------------------|------------|----------|
| JAX     | 1,934.8         | 2,110.6          | **4,045.4** | 1.60×    |
| NumPy   | 2,922.4         | 4,382.7          | **7,305.1** | 2.89×    |
| **Numba**   | **191.2**   | **2,336.8**      | **2,528.0** | 1.00×    |

- **Numba K assembly: NumPy 대비 15.3× 빠름** (2922 / 191 ms)
- **Numba 전체: NumPy 대비 2.9×, JAX 대비 1.6× 빠름**
- 모달 해는 ARPACK 고유해석이 지배 — 백엔드와 무관하게 유사

**주파수 (Modes 1–5):** 24.6 / 61.4 / 61.4 / 98.2 / 122.9 Hz (세 백엔드 동일)

---

## Benchmark — Real Beaded Panel (ccx_iter016)

**Model:** ccx_iter016 비드 패널 (위상 최적화 결과물)  
**Size:** 4,109 nodes / 3,760 elements / 24,654 DOF  
**Solver:** ARPACK, 22 modes  
**C_drill:** 24.9766

| Backend | K Assembly [ms] | Modal Solve [ms] | Total [ms] | vs Numba |
|---------|-----------------|------------------|------------|----------|
| JAX     | 2,040.3         | 4,694.7          | **6,734.9** | 1.48×    |
| NumPy   | 3,877.1         | 7,742.0          | **11,619.1** | 2.55×   |
| **Numba**   | **306.3**   | **4,251.3**      | **4,557.6** | 1.00×    |

- **Numba K assembly: NumPy 대비 12.7× 빠름**
- **Numba 전체: NumPy 대비 2.5×, JAX 대비 1.5× 빠름**

---

## ccx_iter016 vs CalculiX Modal Comparison

**조건:** Numba backend, C_drill=24.9766, fold_alpha=0 (spring 없음)  
**참고:** CalculiX S4 요소는 drilling DOF stiffness 미적용 (MPC 방식 처리)

| Mode | CCX [Hz] | WHT [Hz] | Error |
|------|----------|----------|-------|
| 7    | 4.934    | 4.938    | **+0.07%** |
| 8    | 15.803   | 15.248   | −3.51% |
| 9    | 21.933   | 19.772   | −9.85% |
| 10   | 22.289   | 22.635   | +1.55% |
| 11   | 34.417   | 34.601   | +0.54% |
| 12   | 36.455   | 38.128   | +4.59% |
| 13   | 42.459   | 43.116   | +1.55% |
| 14   | 52.031   | 44.039   | −15.36% |
| 15   | 57.562   | 45.204   | −21.47% |
| 16   | 58.883   | 48.599   | −17.47% |
| 17   | 71.415   | 52.149   | −26.98% |
| 18   | 74.993   | 57.480   | −23.35% |
| 19   | 75.146   | 62.314   | −17.08% |
| 20   | 77.908   | 63.096   | −19.01% |

**MAE (modes 7–20): 11.60% / Max: 26.98%**

**해석 주의:** Mode 14 이후 WHT가 CCX보다 낮은 값(-15~27%)을 보이는 것은  
WHT 스펙트럼에 CCX에 없는 모드가 42~45 Hz 구간에 존재하여  
모드 번호 매칭이 어긋나는 것으로 의심됨 (MAC 분석 필요).

### Numba P_proj 버그 수정 전후 비교 (Mode 7)

| 조건 | Numba | JAX | vs CCX |
|------|-------|-----|--------|
| 수정 전 (pz=0) | 6.9980 Hz | 4.9376 Hz | Numba: **+42%** |
| **수정 후 (3D pz)** | **4.9376 Hz** | **4.9376 Hz** | **+0.07%** |

---

## Fold-Line Hinge Spring (fold_alpha=0.5, ccx_iter016)

| 조건 | MAE vs CCX | Max Error |
|------|-----------|-----------|
| 기본 (spring 없음) | 11.60% | 26.98% |
| fold_alpha=0.5 | 11.60% | 26.96% |

비드 패널에서는 fold spring 효과 미미 (MAE 변화 -0.01 pp).  
단순 V-beam 테스트에서는 α=0.5 적용 시 오차 −2.61% → −0.16%로 개선.

---

## Test Environment

| Item | Value |
|------|-------|
| Python | 3.x |
| NumPy | standard |
| Numba | nopython JIT |
| JAX | vmap/jit (CPU) |
| Solver | ARPACK (scipy.sparse.linalg.eigsh) |
| Mesh (patch) | 441 nodes, 400 QUAD4 / 800 TRIA3 elements |
| Total DOF (patch) | 2,646 (2,563 free after BC) |
| Plate geometry | 200 × 200 mm, t=1 mm |
| Material | E=210 GPa, ν=0.3, ρ=7.85×10⁻⁹ ton/mm³ |
| C_drill | 24.9766 (CalculiX Mode 7 교정값) |
