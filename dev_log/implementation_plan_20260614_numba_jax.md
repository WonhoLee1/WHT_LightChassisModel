# Numba 및 JAX 기반 응력 복원 엔진 구현 및 성능 벤치마크 계획

현재 성공적으로 최적화된 Numpy 기반의 `ElementStressRecovery`를 바탕으로, 극한의 성능 최적화를 위한 **Numba 버전**과 **JAX 버전**을 각각 분리하여 구현하고, 결과를 검증 및 벤치마크합니다.

## ⚠️ User Review Required

본 구현 계획에 대해 확인 후 승인해 주시면 작업(코딩)을 시작합니다.
- 파일 구조 분리 여부: 기존 코드를 안전하게 보존하기 위해 별도 파일(`wht_stress_recovery_numba.py`, `wht_stress_recovery_jax.py`)로 구현하는 방안을 제안합니다. 동의하시나요?

## 📝 Proposed Changes

### 1. Numba 버전 구현 (`wht_solver/wht_stress_recovery_numba.py`)
- **클래스명**: `ElementStressRecoveryNumba`
- **전략**: 
  - Python 객체(Dict, 클래스 등)는 Numba의 `@njit` (No-Python 모드)과 호환되지 않습니다.
  - 따라서 파이썬 래퍼 클래스가 데이터를 순수 Numpy Array(스칼라/배열)로 풀어서 준비한 뒤, 핵심 수학 연산을 담당하는 `@njit(parallel=True)` 함수(커널)로 넘기는 구조로 작성합니다.
  - `np.einsum`이나 `np.outer` 같이 Numba에서 성능 저하를 유발하거나 지원하지 않는 함수는 `prange`를 활용한 C 스타일의 명시적 다중 루프로 재작성하여 **메모리 할당을 최소화**하고 멀티스레드 성능을 극대화합니다.

### 2. JAX 버전 구현 (`wht_solver/wht_stress_recovery_jax.py`)
- **클래스명**: `ElementStressRecoveryJax`
- **전략**:
  - 기존 Numpy 기반 벡터화 코드와 가장 유사하게 작성하되, `import jax.numpy as jnp`를 사용합니다.
  - JAX의 특성상 배열 내부 요소 수정(`arr[0] = x`)이 불가능하므로, `jnp.zeros` 대신 `jnp.stack`과 `jnp.concatenate`를 사용하여 행렬과 Voigt 텐서를 조립하는 함수형(Functional) 스타일로 코드를 수정합니다.
  - 최하단 연산 로직(`_compute_at_z` 및 `_compute_all_layers` 상당)에 `@jax.jit` 컴파일러를 씌워 **커널 퓨전(Kernel Fusion)**을 유도하고, 반복 연산을 하나로 압축합니다.

### 3. 검증 및 벤치마크 스크립트 (`test_jaxSSO/benchmark_stress_recovery.py`)
- **기능**: 세 가지 솔버(Numpy, Numba, JAX)를 동일한 입력(절점 변위 데이터 및 섀시 모델)으로 구동합니다.
- **결과 비교(Accuracy)**: Numpy 버전을 Ground Truth(정답)로 삼고, `np.allclose(numpy_res, numba_res, atol=1e-5)` 등을 통해 계산 결과의 정합성을 수치적으로 검증합니다.
- **속도 비교(Performance)**: JAX의 JIT 컴파일 시간(Warm-up)과 실제 순수 연산 시간(Inference)을 구분하여 측정하며, Numpy/Numba/JAX 3자의 161프레임 환산 처리 속도(초 및 FPS)를 터미널에 명확히 출력합니다.

## 🔍 Verification Plan

1. **테스트 스크립트 작성**: 섀시 모달 해석 결과 데이터를 로딩하여 단일 프레임 변위를 주입하는 스크립트를 작성합니다.
2. **정합성(Accuracy) 통과**: Numpy 결과물과 Numba, JAX의 결과물(Stress, Strain) 배열 오차가 1e-5 이하인지 확인합니다.
3. **성능 프로파일링**: 3가지 엔진에 대해 100회씩 연산을 반복 수행하여 평균 실행 시간을 도출하고, "Numpy vs Numba vs JAX" 벤치마크 결과표를 제공합니다.
