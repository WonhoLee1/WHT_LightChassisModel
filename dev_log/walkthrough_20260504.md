# Shell 응력/변형률 Through-Thickness 적분점 개선 Walkthrough

## 날짜: 2026-05-04

## 개요
MITC4/MITC3 Shell 요소의 응력/변형률 계산 파이프라인을 상용 CAE S/W(OptiStruct, Nastran) 수준으로 고도화하였습니다.

## 변경 사항

### 1. `wht_stress_recovery.py` — 전면 리팩토링

**Before:** 
- 두께 적분점 없음 (z = +t/2 하드코딩)
- QUAD4/TRIA3 각각 중복 코드
- 반환: 4개 tuple `(stress, strain_total, strain_membrane, strain_bending)`

**After:**
- 3개 적분점: Upper (+t/2), Mid (0), Lower (-t/2)
- 공통 계산 함수 `_compute_at_z()` 추출 → 코드 중복 완전 제거
- Max Envelope (Upper/Lower 중 Von Mises 최대)
- Membrane/Bending 분리 (Stress에도 적용)
- 반환: `Dict[str, np.ndarray]` 12개 키

### 2. `wht_solver.py` — API 통합

- `solve_static()`: 새 dict API 사용, 12개 cell_data 키 자동 생성
- `solve_modal()`: 기존 호환 유지 (Upper surface 기본)

### 3. `wht_visualizer.py` — Shell Layer UI 추가

- **Shell Layer 콤보박스**: Category/Component 행 아래에 배치
  - 옵션: Upper (+t/2), Mid (0), Lower (-t/2), Max Envelope, Membrane, Bending
  - Stress/Strain 카테고리에서만 활성화
- **카테고리 필터링**: Shell Layer 변형 카테고리가 별도 항목으로 등장하지 않도록 필터링
- **Signed VonMises, 3D Principal 지원 유지**

### 4. `patch_test_extended.py` — 테스트 확장 (8/8 PASS)

| Test | 내용 | 결과 |
|------|------|------|
| Test 4 | 45° 회전 QUAD4 인장 → 전역 텐서 회전 | ✅ PASS |
| Test 5 | TRIA3 순수 인장 | ✅ PASS |
| Test 6 | TRIA3 순수 전단 | ✅ PASS |
| Test 7 | TRIA3 순수 굽힘 | ✅ PASS |
| Test 8 | QUAD4 인장+굽힘 복합 | ✅ PASS |
| Test 9 | 순수 굽힘 → Mid-plane 응력 = 0 | ✅ PASS |
| Test 10 | 순수 굽힘 → Upper = -Lower 대칭성 | ✅ PASS |
| Test 11 | 순수 인장 → Upper = Mid = Lower | ✅ PASS |

## 기술적 핵심 사항

### 두께 방향 응력 계산 원리

Shell 요소의 변형률은 Membrane + Bending으로 분해됩니다:

```
ε(z) = ε_membrane + z · κ
```

- **Membrane (z=0)**: 면내 변형만. 굽힘 없는 순수 인장/압축/전단.
- **Bending (z=±t/2)**: 곡률에 의한 변형. z에 선형 비례.
- **Total (z=+t/2)**: Upper surface의 최종 응력. 기존 출력과 동일.

### Max Envelope 계산

각 요소별로 Upper/Lower의 Von Mises를 비교하여 큰 쪽의 전체 텐서를 선택합니다.
이는 상용 S/W의 "Worst Case" 표시와 동일한 기능입니다.

## 백업 파일

- `wht_solver/wht_stress_recovery_backup_20260504.py`: 리팩토링 전 원본
