import jax
import jax.numpy as jnp
from typing import List, Optional

class ModeTracker:
    """
    [모드 트래커] MAC(Modal Assurance Criterion)을 통해 특정 모드를 식별합니다.
    JAX 미분 루프 내에서 Side-effect를 방지하기 위해 상태 업데이트를 수행하지 않는 Pure Function들을 제공합니다.
    """
    def __init__(self, target_mode_idx: int = 0):
        self.target_mode_idx = target_mode_idx
        
    @staticmethod
    def calculate_mac(mode_a: jnp.ndarray, mode_b: jnp.ndarray) -> jnp.ndarray:
        numerator = jnp.abs(jnp.dot(mode_a, mode_b))**2
        denominator = jnp.dot(mode_a, mode_a) * jnp.dot(mode_b, mode_b)
        return numerator / (denominator + 1e-10)

    def find_best_match(self, current_modes: jnp.ndarray, ref_mode: jnp.ndarray) -> int:
        """ 현재 모드들 중 기준 모드와 가장 유사한 모드 인덱스 반환 (Pure Function) """
        mac_values = jax.vmap(lambda m: self.calculate_mac(m, ref_mode))(current_modes)
        return jnp.argmax(mac_values)

class DynamicConstraint:
    """
    [동적 제약 조건] 특정 모드의 고유 진동수를 제어합니다.
    """
    def __init__(self, target_freq: float, mode_idx: int = 0, penalty_weight: float = 100.0):
        self.target_freq = target_freq
        self.tracker = ModeTracker(target_mode_idx=mode_idx)
        self.weight = penalty_weight
        
    def get_penalty(self, freqs: jnp.ndarray, modes: jnp.ndarray, ref_mode: jnp.ndarray) -> float:
        """
        Pure Function: 외부에서 ref_mode를 전달받아 페널티 계산
        """
        idx = self.tracker.find_best_match(modes, ref_mode)
        current_f = freqs[idx]
        violation = jnp.maximum(0.0, self.target_freq - current_f)
        return self.weight * (violation**2)

class StressConstraint:
    """
    [응력 제약 조건] 특정 부위의 최대 응력을 제한합니다.
    """
    def __init__(self, limit_stress: float):
        self.limit = limit_stress

if __name__ == "__main__":
    pass
