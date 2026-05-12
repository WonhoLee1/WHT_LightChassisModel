# WHT LightChassisModel Issue Tracker

이 파일은 프로젝트의 개선 사항, 요청 작업, 미해결 문제 등을 관리하기 위한 파일입니다. (규칙 4 준수)

## 🛠️ 진행 중인 이슈 (Ongoing Issues)

- [x] **[시각화]** `wht_visualizer.py`의 기본 메시지 에지 색상을 'dark gray'로 통일 (완료)
- [x] **[코드 가독성]** `exam2_shell_jaxSSO_load.py`의 Docstring 상세화 및 한글화 (완료)
- [x] **[솔버]** RBE3 보간 요소(Interpolation Element) 구현 및 응력 집중 문제 완화 (2026-05-03)
- [x] **[모델링]** 경로 추적(Path Tracing) 및 플러드 필(Flood Fill) 기반 고급 플랜지 노드 선택 로직 구현 (2026-05-03)
- [x] **[최적화]** 정밀 FEA 기반 멀티 케이스(하중별 BC 분리) 토포그래피 최적화 솔버 구현 (2026-05-05)
- [x] **[설계 수정]** Topology(SIMP 밀도) → Topography(노드 비드 높이) 설계 변수 전환: `WHTopographySolver` 재설계 (2026-05-05)
- [x] **[동역학]** 실측 CSV 위치 데이터 기반 고정밀 동적 응답 해석(SPCD) 파이프라인 통합 (2026-05-12)
    - [x] Nodal BC 및 RBE3 마스터 노드 제어 로직 적용
    - [x] 회전 자유도 Free 상태 유지 및 t-start 필터링 옵션 구현
    - [x] JAX 솔버 인덱싱 및 셀 카운트 검증 오류 해결


## ✅ 완료된 이슈 (Completed Issues)

- [x] **[인프라]** `issue_tracker.md` 초기화 (2026-05-03)
- [x] **[로깅]** `dev_log`에 코드 점검 결과 기록 (2026-05-03)

## 📌 향후 개선 사항 (Backlog)

- [ ] 전체 모듈에 대해 사용자 규칙(Korean Docstring, OOP 등) 준수 여부 재검토.
- [ ] PyVista 뷰어의 컨텍스트 메뉴 기능 확장 (스크린샷 저장 경로 설정 등).
