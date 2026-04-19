# WHT_LightChassisModel Issue Tracker

## [2026-04-18] JaxSSO Modal Analysis API Compatibility
- **이슈**: `exam1_nf.py` 실행 시 `Model` 객체에 `known_id`, `unknown_id` 속성이 없어 `AttributeError` 발생.
- **원인**: JaxSSO 최신 버전에서는 `known_indices`만 제공하며, 자유도(DOF) 분리 로직이 변경됨.
- **해결**: 
    - `known_id = model.known_indices`로 수정.
    - `unknown_id = np.setdiff1d(np.arange(ndof), known_id)`로 자유 DOF 직접 계산.
- **결과**: 고유진동수 해석 정상 수행 완료 (1차 모드: 215.53 Hz).
- **상태**: 완료 (Resolved)

## [2026-04-19] HEX27 Rollback & HEX8 Refinement
- **이슈**: HEX27(고차 요소) 도입 시 JAX 커널 컴파일 및 메모리 사용량이 과도하게 높음.
- **결정**: HEX27 지원을 원복하고, 표준 HEX8(선형 요소)을 사용하되 두께 방향 요소 분할(`wall_layers`)을 강화하여 전단 잠김(Shear Locking) 해결 시도.
- **조치**: 
    - `mesh_utils.py`: HEX27 업그레이드 및 27노드 재정렬 로직 제거.
    - `exam2_solid_jaxfem.py`: 요소 타입을 HEX8로 변경하고 `wall_layers=3` 적용.
- **결과**: (진행 중) HEX8 기반 수렴성 확인 필요.
- **상태**: 진행 중 (In Progress)
