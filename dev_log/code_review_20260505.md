# wht_topo 코드 리뷰 및 리팩토링 (2026-05-05)

## 개요
`wht_topo` 패키지 전체(5개 파일)에 대한 정밀 코드 리뷰를 수행하고,
발견된 결함을 심각도별로 분류하여 전수 수정하였습니다.

## 수정 내역

### 🔴 Critical (5건) — 런타임 에러 유발
| ID | 파일 | 내용 | 수정 |
|:--:|:--:|:---|:---|
| C1 | solver.py:29 | `Tuple` 미임포트 → `NameError` | `from typing import Tuple` 추가 |
| C2 | solver.py:100 | `self.base_stiffness` 미정의 → `AttributeError` | `_precompute_geometric_data`에서 정의 |
| C3 | solver.py:116 | `solve_physics_jit` 시그니처 불일치 | 모듈 레벨 순수 함수로 분리, 인자 통일 |
| C4 | solver.py:83 | `load_tensors` 딕셔너리 빈 값 | `_precompute_load_sensitivities`로 재설계 |
| C5 | solver.py:137 | 반환값 언패킹 불일치 | 호출부와 정의부 일치시킴 |

### 🟠 Major (5건) — 논리적 오류
| ID | 파일 | 내용 | 수정 |
|:--:|:--:|:---|:---|
| M1 | solver.py:119 | 목적 함수에서 Twisting/Lifting 누락 | 3개 하중 모두 가중 합산 |
| M2 | solver.py:159 | MAC 모드 트래킹 코드 삭제됨 | 전체 복원 |
| M3 | mma.py:44 | vol_limit 0.3 하드코딩 | 생성자에서 `vol_frac` 주입 |
| M4 | mma.py:50 | `sqrt(음수)` → NaN 전파 | `safe_df0dx = min(df0dx, -1e-10)` |
| M5 | loads.py:163 | Lifting 랜덤 코너 → 비결정적 감도 | 결정적 `corner_idx` 인자로 변경 |

### 🟡 Minor (5건) — 코드 위생
| ID | 파일 | 내용 | 수정 |
|:--:|:--:|:---|:---|
| m1 | run_topo.py:29 | Mutable default `draw_dir=[]` | `None` 패턴으로 교체 |
| m2 | run_topo.py:23 | 미사용 `ModeTracker` 임포트 | 제거 |
| m3 | solver.py:97 | `@jax.jit` on bound method | 모듈 레벨 함수로 분리 |
| m4 | constraints.py:33 | Dead code `self.ref_mode` | 제거 |
| m5 | solver.py 전반 | Docstring 부재 | 전체 추가 |

## 아키텍처 개선
- `solve_physics_jit` → 모듈 레벨 `_compute_compliance_and_freq` 순수 함수로 분리
- `_heaviside` → `@staticmethod`로 추출하여 중복 인라인 제거
- `MMAOptimizer` → `vol_frac`를 생성자에서 주입받도록 인터페이스 확장
