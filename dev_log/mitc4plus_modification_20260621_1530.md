# MITC4+ Membrane Tying 구현 기록

**날짜**: 2026-06-21  
**대상 파일**: `wht_solver/wht_quad4_element.py`, `wht_solver/wht_quad4_element_jax.py`

---

## 1. 수정 배경 및 동기

`wht_quad4_element.py`는 파일 헤더에 "MITC4+ High-Fidelity Shell Element"라고 명시되어 있으나,
실제 막(membrane) 강성은 표준 2×2 가우스 적분을 사용하고 있었다.

- **기존 방식**: 전체 가우스 적분 (Km_full) + 강체 투영(P_proj)으로 막 잠금(membrane locking) 방지  
  (SRI 코드는 `Km_1pt`를 계산하나 실제로는 사용하지 않는 dead code였음)
- **MYSTRAN MITC4+**: `PARAM,QUAD4TYP,MITC4+` 설정 시 Ko-Lee-Bathe 2016 5-point membrane tying 적용

MYSTRAN vs WHT 비교(twisting 케이스):
- MYSTRAN MITC4+(구현 전): 3725 mm
- WHT: 2917 mm (약 -22%)

MITC4+ membrane tying 도입으로 왜곡 요소에서의 정확도 향상이 기대된다.

---

## 2. 이론 요약 (Ko, Lee, Bathe 2016)

### 특성 기하 벡터 (Eqn 11)

노드 파라메트릭 좌표: node 0=(-1,-1), 1=(+1,-1), 2=(+1,+1), 3=(-1,+1)

```
X_R = (1/4) Σ ξᵢ xᵢ  = (1/4)(-x0 + x1 + x2 - x3)
X_S = (1/4) Σ ηᵢ xᵢ  = (1/4)(-x0 - x1 + x2 + x3)
X_D = (1/4) Σ ξᵢηᵢ xᵢ = (1/4)(x0 - x1 + x2 - x3)   ← 왜곡 벡터
```

### 쌍대 기저 및 왜곡 변수 (Eqn 11, 24)

```
det_RS = X_R[0]*X_S[1] - X_R[1]*X_S[0]
c_r = (X_D × X_S)_z / det_RS = (X_D[0]*X_S[1] - X_D[1]*X_S[0]) / det_RS
c_s = (X_R × X_D)_z / det_RS = (X_D[1]*X_R[0] - X_D[0]*X_R[1]) / det_RS
d   = c_r² + c_s² - 1
```

직사각형 요소: X_D = 0, c_r = c_s = 0, d = -1 (공식 정상 적용)  
특이점 조건: |d| < 1e-10 → 표준 이소파라메트릭 Bm으로 폴백

### 5-point Membrane Tying (Eqn 27a/b/c)

타잉점: A=(0,1), B=(0,-1), C=(1,0), D=(-1,0), E=(0,0)

가중치:
```
a_A = c_r(c_r-1)/(2d),  a_B = c_r(c_r+1)/(2d)
a_C = c_s(c_s-1)/(2d),  a_D = c_s(c_s+1)/(2d)
a_E = 2*c_r*c_s / d
```

조립 (R=ξ_g, S=η_g):
```
Bm[exx] = 0.5(1-2a_A+S+2a_A S²) BmA_exx + 0.5(1-2a_B-S+2a_B S²) BmB_exx
         + a_C(-1+S²) BmC_eyy + a_D(-1+S²) BmD_eyy + a_E(-1+S²) BmE_exy

Bm[eyy] = a_A(-1+R²) BmA_exx + a_B(-1+R²) BmB_exx
         + 0.5(1-2a_C+R+2a_C R²) BmC_eyy + 0.5(1-2a_D-R+2a_D R²) BmD_eyy
         + a_E(-1+R²) BmE_exy

Bm[exy] = 0.25(R+4a_A RS) BmA_exx + 0.25(-R+4a_B RS) BmB_exx
         + 0.25(S+4a_C RS) BmC_eyy + 0.25(-S+4a_D RS) BmD_eyy
         + (1+a_E RS) BmE_exy
```

여기서 BmX_eYY는 타잉점 X에서 계산한 표준 B-행렬의 eYY 행 벡터 (24,).

---

## 3. 특이점 처리

| 조건 | 처리 |
|------|------|
| `abs(d) > 1e-10` AND `abs(det_RS) > 1e-12` | MITC4+ 5-point tying |
| 그 외 (직사각형에 가까운 요소, 또는 퇴화) | 표준 이소파라메트릭 Bm |

직사각형 요소는 d=-1이므로 항상 MITC4+가 적용됨.  
공식 자체가 d=-1일 때 정확히 MITC4 membrane tying(edge 4-point 보간)으로 귀결됨.

---

## 4. 수정된 함수 목록

### wht_solver/wht_quad4_element.py

**새로 추가된 함수**:
- `_nb_dN_dx_dy_only(xi, eta, coords)` — dN/dx, dN/dy만 반환 (Numba JIT)
- `_nb_Bm_rows_at(xi, eta, coords)` — exx, eyy, exy 행 벡터 반환 (Numba JIT)
- `_nb_Bm_mitc4plus_assemble(R, S, BmA_exx, ...)` — Eqn 27 조립 (Numba JIT)

**수정된 함수**:
- `_element_K_mitc4_plus_nb()`:
  - 기존 SRI 사전계산 (dead code) 제거
  - MITC4+ 특성벡터·타잉점 사전계산 블록 추가
  - Gauss loop 내 `Bm` 산출을 MITC4+ 또는 표준 B로 분기
- `_element_K_mitc4_plus_np()`:
  - 동일 변경 (NumPy 폴백 버전)

### wht_solver/wht_quad4_element_jax.py

**수정된 함수**:
- `_element_K_mitc4_plus_jax()`:
  - SRI dead code 제거
  - MITC4+ 사전계산 추가
  - `jnp.where(use_m4p, Bm_m4p, Bm_std)`로 jit 호환 분기

**변경 없음**: 전단 타잉(shear MITC, Bs13/Bs23), 드릴링 안정화(K_drill), 강체 투영(P_proj)

---

## 5. 검증 결과

### 단일 요소 패치 테스트 (t=1mm, E=210GPa, ν=0.3)

| 항목 | NumPy | Numba | JAX |
|------|-------|-------|-----|
| 직사각 대칭성 오차 | 1.16e-10 | 1.16e-10 | 2.33e-10 |
| 직사각 최소 6 고유값 | ~0 (1e-8 이하) | ~0 | ~0 |
| 직사각 7번째 고유값 | 13.4 (양수) | 13.4 | 13.4 |
| 왜곡요소 대칭성 오차 | 3.73e-9 | 3.73e-9 | 3.73e-9 |
| 왜곡요소 최소 6 고유값 | ~0 (1e-8 이하) | ~0 | ~0 |

- 대칭성: float64 기계 정밀도(~1e-15) 대비 허용 범위 내
- 강체 모드 6개 모두 0에 수렴
- 양의 탄성 고유값 확인 → 요소 물리적 타당성 검증

---

## 6. 기대 효과

- 왜곡된 사각형 요소(non-rectangular quad)에서 막 잠금 제거 개선
- WHT와 MYSTRAN MITC4+ 간 twisting 변위 차이 감소 기대
  - 현재: WHT 2917mm vs MYSTRAN(MITC4+) 3725mm (+27% 차이)
  - 변경 후: 차이 감소 예상 (요소 왜곡 정도에 따라 다름)
- 벤딩 응답은 동일 (Bb 미변경)
- 전단 응답은 동일 (MITC shear tying 미변경)
