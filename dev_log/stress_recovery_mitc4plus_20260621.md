# Stress Recovery MITC4+ gamma_xy 수정 기록

**날짜**: 2026-06-21  
**대상 파일**: `wht_solver/wht_stress_recovery.py`  
**관련 파일**: `wht_solver/wht_quad4_element.py` (MITC4+ membrane tying — `mitc4plus_modification_20260621_1530.md` 참조)

---

## 1. 문제 발생 경위

MITC4+ membrane tying 적용 후 (wht_quad4_element.py) 다음 모순이 발견됨:
- twisting 변위: 2917 → 3436mm (+18%) — 올바른 방향
- twisting Median Von Mises: 342 → **296 MPa** (-13%) — **물리적으로 불가능** (변위 증가 시 응력도 증가해야 함)

MYSTRAN MITC4+ 대비: WHT 296 MPa vs MYSTRAN 610 MPa (2배 차이)

---

## 2. 원인 진단

`wht_stress_recovery.py`의 `recover_quad4` 함수 line 160-192:

```python
# 중심점(xi=0, eta=0) 표준 isoparametric — 수정 전
dNxi  = np.array([-1.0, 1.0, 1.0, -1.0]) * 0.25
dNeta = np.array([-1.0, -1.0, 1.0, 1.0]) * 0.25
# ... Jacobian 계산 ...
dN_dx = outer(invJ11, dNxi) + outer(invJ12, dNeta)   # (M,4)
dN_dy = outer(invJ21, dNxi) + outer(invJ22, dNeta)   # (M,4)

gamma_xy_m = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)  # 표준 B_m exy 행
```

K 조립은 MITC4+ B_m(tying 기반)을 쓰지만, 응력 복원은 표준 B_m을 사용 → **B 행렬 불일치**.  
특히 `gamma_xy`의 Eqn 27c (5개 타잉점 조합)가 왜곡 요소에서 표준 B_m과 크게 달라짐.

---

## 3. 수정 내용

### 추가 수정 1: Numba 캐시 삭제

MITC4+ 코드 변경 후 `__pycache__`에 이전 버전 캐시가 남아있어 변경이 반영되지 않았음:
```
wht_solver/__pycache__/wht_quad4_element._element_K_mitc4_plus_nb-345.py312.*
wht_solver/__pycache__/wht_quad4_element._nb_Bm_Bb-248.py312.*
...
```
→ `wht_solver/__pycache__/wht_quad4_element*` 전체 삭제

### 추가 수정 2: gamma_xy_m MITC4+ 교체

**Ko-Lee-Bathe 2016 Eqn 27c** 기반 gamma_xy 계산:

#### 특성 기하 벡터 (요소별 2D 로컬 좌표)
```python
X_R = 0.25*(-C_loc[:,0] + C_loc[:,1] + C_loc[:,2] - C_loc[:,3])  # (M,2)
X_S = 0.25*(-C_loc[:,0] - C_loc[:,1] + C_loc[:,2] + C_loc[:,3])
X_D = 0.25*(+C_loc[:,0] - C_loc[:,1] + C_loc[:,2] - C_loc[:,3])

det_RS = X_R[:,0]*X_S[:,1] - X_R[:,1]*X_S[:,0]
c_r = (X_D[:,0]*X_S[:,1] - X_D[:,1]*X_S[:,0]) / det_RS
c_s = (X_D[:,1]*X_R[:,0] - X_D[:,0]*X_R[:,1]) / det_RS
d   = c_r**2 + c_s**2 - 1
```

#### 가중치 (요소별 스칼라)
```python
a_A = c_r*(c_r-1)/(2*d),  a_B = c_r*(c_r+1)/(2*d)
a_C = c_s*(c_s-1)/(2*d),  a_D = c_s*(c_s+1)/(2*d)
a_E = 2*c_r*c_s/d
```

#### 타잉점 5개에서 표준 gamma_xy 계산 후 Eqn 27c 조합
```python
# 타잉점 (xi, eta): A=(0,1), B=(0,-1), C=(1,0), D=(-1,0), E=(0,0)
gxy_P = np.sum(dN_dy_P * u_x + dN_dx_P * u_y, axis=1)  # for each P

# 2×2 Gauss 평균 적용
gxy_mitc4p = 0  # sum over 4 gauss points
for R, S in gauss_pts:
    gxy_mitc4p += (
        0.25*(R + 4*a_A*R*S) * gxy_A
      + 0.25*(-R + 4*a_B*R*S) * gxy_B
      + 0.25*(S + 4*a_C*R*S) * gxy_C
      + 0.25*(-S + 4*a_D*R*S) * gxy_D
      + (1 + a_E*R*S) * gxy_E
    )
gamma_xy_m = gxy_mitc4p / 4.0
```

#### 특이점 폴백
```python
valid = (np.abs(d) > 1e-10) & (np.abs(det_RS) > 1e-12)
gamma_xy_m = np.where(valid, gamma_xy_m_mitc4p, gamma_xy_m_std)
```

---

## 4. 검증 결과

단일 요소 패치 테스트 + 전체 모델 실행:

| 케이스 | max_disp | Median VM (수정 전 → 후) |
|--------|----------|--------------------------|
| bending | 454.5mm | 135.30 MPa → 135.30 MPa (**변화 없음** ✓) |
| twisting | 3435.8mm | 296 MPa → **342 MPa** (+15%) |

- bending 불변: σ_xx, σ_yy 지배 케이스이므로 gamma_xy 변경 영향 없음
- twisting 개선: 296 → 342 MPa (+15%)

---

## 5. 잔여 차이 분석

| 항목 | WHT MITC4+ | MYSTRAN MITC4+ |
|------|-----------|----------------|
| twisting 변위 | 3436mm | 3725mm (–8%) |
| twisting Median VM | 342 MPa | 610 MPa (–44%) |

**잔여 차이 원인**:
1. **응력 복원 위치**: WHT는 centroid 1점, MYSTRAN은 corner 외삽(`STRESS(PRINT,CORNER)`) — corner 외삽은 응력 집중 영역에서 더 높은 값 출력
2. **3D 공변 기저 처리**: MYSTRAN은 완전 3D 쉘 공변 변환, WHT는 로컬 2D 평면 투영 방식
3. **변위 차이**: 3436 vs 3725mm (8%) → 비례 응력 차이 ~8% 기여

Max Von Mises (22616 MPa) 및 모달 주파수는 일치하므로 전체 하중 전달 메커니즘은 동일.

---

## 6. 향후 개선 여지

- `recover_quad4_nodal`에도 동일한 MITC4+ gamma_xy 적용 (현재 미적용)
- Corner 외삽 응력 복원 강화로 MYSTRAN CORNER 출력과 공정 비교 가능
