# 솔버 검증 및 패치 테스트(SolverVerification) 구축 계획 (2026-04-20)

현재 WHTSolver의 쉘 요소(QUAD4, TRIA3) 성능을 검증하고, 고유진동수가 비정상적으로 높게 나오는 원인을 분석하기 위해 표준 패치 테스트 및 벤치마크 스위트를 구축합니다.

## 제안된 변경 사항

### [SolverVerification] 신규 폴더 생성

1. **analytical_solutions.py**: Kirchhoff 판 이론 및 보 이론에 기반한 해석적 해 제공.
2. **patch_tests.py**: `WHTMeshModel`과 `WHTSolver`를 사용하는 테스트 케이스 구현. (3점 굽힘, 인장 패치, 고유진동수 등)
3. **verification_runner.py**: 모든 테스트를 실행하고 테이블 형식으로 출력.

## 검증 계획
- `python .\SolverVerification\verification_runner.py` 실행 및 오차율 5% 이내 확인.
