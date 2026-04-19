# OpenRadioss Automation Challenge: 10g Sine Wave Analysis

이 계획은 `exam1_nf.py`에서 사용된 GMSH 트레이 모델을 OpenRadioss용 데이터로 변환하고, 10g Sine 가속도 경계 조건을 부여하여 해석을 수행하는 자동화 프로세스를 구축하는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> **OpenRadioss 2025+**: 최신 공개 버전인 2025 규격을 준수하며, `/VERS/2025` 헤더를 포함합니다.
> **카드 포맷 준수**: 고정폭(Fixed-width, 10자 단위 등) 형식을 철저히 지켜 `.rad` 파일을 생성합니다.
> **OpenRadioss 설치 경로**: `D:\OpenRadioss_win64\OpenRadioss\exec` 경로를 기반으로 환경 변수를 설정합니다.

## Proposed Changes

### 1. 전처리 및 모델 생성 (Preprocessing)

#### [NEW] [radioss_challenge.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/radioss_challenge.py)
`exam1_nf.py`의 로직을 활용하여 GMSH 메시를 생성하고, 이를 OpenRadioss 포맷으로 작성하는 메인 스크립트입니다.
- **Mesh Generation**: `mesh_utils.generate_shell_tray`를 호출하여 노드 및 요소 정보를 획득합니다.
- **Radioss Writer**: 2025 규격에 맞춰 고정폭 필드(10자 단위 등)로 `/NODE`, `/SHELL`, `/PART`, `/MAT/LAW1`, `/PROP/SHELL` 카드를 생성합니다.
- **Boundary Conditions (BC)**: 
    - Rim 노드들을 추출하여 `/SET/NODE`로 정의합니다.
    - `/IMPBACC`를 사용하여 해당 세트에 Z방향 가속도를 부여합니다.
    - `/FUNCT`를 정의하여 10g Sine 파형(Amplitude: 98066.5 mm/s²)을 기술합니다.

### 2. 솔버 실행 엔진 (Solver Execution)

#### [MODIFY] [radioss_challenge.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/radioss_challenge.py) (내부 메서드)
OpenRadioss 솔버를 호출하는 로직을 포함합니다.
- **Environment Setup**: `OPENRADIOSS_PATH`, `RAD_H3D_PATH` 등 필요한 환경 변수를 설정합니다.
- **Starter Run**: `starter_win64.exe -i <file>.rad`
- **Engine Run**: `engine_win64.exe -i <file>.rad`
- **Error Handling**: 로그 파일(`.out`)을 파싱하여 Normal Termination 여부를 확인합니다.

### 3. 후처리 (Post-processing)

#### [NEW] [process_results.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/process_results.py) (또는 챌린지 스크립트 내 포함)
사용자가 언급한 도구들을 사용하여 결과를 확인합니다.
- **TH to CSV**: `th_to_csv_win64.exe`를 사용하여 시간 이력 결과(`T01`)를 CSV로 변환하고, 에너지 보존 법칙(Energy Error)과 가속도 추종성을 확인합니다.
- **Anim to VTK**: `anim_to_vtk_win64.exe`를 사용하여 애니메이션 결과를 VTK로 변환하여 시각화 준비를 마칩니다.

## Open Questions

- **Sine파 주기**: 0.01초 내에 몇 주기의 Sine파를 넣길 원하시나요? (현재 1주기로 가정)
- **결과 시각화**: 해석 완료 후 PyVista나 ParaView로 결과를 자동 렌더링하는 과정을 포함할까요?

## Verification Plan

### Automated Tests
1. **Mesh Validation**: 생성된 `.rad` 파일의 노드 및 요소 개수가 GMSH 출력과 일치하는지 확인.
2. **Solver Run**: `starter`와 `engine` 실행 결과 에러 코드 0 반환 및 `.out` 파일 내 "NORMAL TERMINATION" 문자열 확인.
3. **Data Integrity**: 추출된 CSV 파일에서 시간에 따른 변위/가속도 값이 Sine 파형을 따르는지 확인.

### Manual Verification
- ParaView를 통해 변환된 `.vtk` 파일을 열어 Tray의 동적 거동 확인.
