import time
import numpy as np
import sys
import os

# JAX Warning suppression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("JAX is not installed. Exiting.")
    sys.exit(1)

try:
    import numba
except ImportError:
    print("Numba is not installed. Exiting.")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wht_solver.wht_stress_recovery import ElementStressRecovery
from wht_solver.wht_stress_recovery_jax import ElementStressRecoveryJax
from wht_solver.wht_stress_recovery_numba import ElementStressRecoveryNumba

class MockElement:
    def __init__(self, nids):
        self.node_ids = nids
        self.type = "QUAD4" if len(nids) == 4 else "TRIA3"
        self.pid = 1

class MockNode:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockProp:
    t = 1.2
    mid = 1

class MockMat:
    E = 210000.0
    nu = 0.3

class MockModel:
    def __init__(self, n_elements=5000):
        self.elements = {}
        self.nodes = {}
        self.properties = {1: MockProp()}
        self.materials = {1: MockMat()}
        self.element_types = {}
        
        # Grid of nodes
        grid_w = int(np.sqrt(n_elements)) + 2
        for i in range(n_elements + grid_w + 2):
            self.nodes[i] = MockNode(float(i % grid_w), float(i // grid_w), 0.0)
            
        # Create elements
        for i in range(n_elements):
            n1 = i
            n2 = i + 1
            n3 = i + grid_w + 1
            n4 = i + grid_w
            self.elements[i+1] = MockElement([n1, n2, n3, n4]) # Proper quad

def run_benchmark():
    print("Initializing Mock Model (N=5000 elements)...")
    M = 5000
    model = MockModel(M)
    
    # 161 frames
    frames = 161
    u_global_frames = np.random.randn(frames, len(model.nodes), 6) * 0.01
    
    sorted_nids = list(model.nodes.keys())
    
    fields = ["Stress", "Strain", "Stress (Max Envelope)"]
    
    print("\n--- 1. NumPy (Baseline) ---")
    start = time.perf_counter()
    # Warmup
    _ = ElementStressRecovery.recover_quad4_nodal(model, u_global_frames[0], sorted_nids, fields=fields)
    numpy_time_start = time.perf_counter()
    for f in range(frames):
        res_np = ElementStressRecovery.recover_quad4_nodal(model, u_global_frames[f], sorted_nids, fields=fields)
    numpy_time = time.perf_counter() - numpy_time_start
    print(f"NumPy Time for {frames} frames: {numpy_time:.4f}s ({frames/numpy_time:.2f} FPS)")

    print("\n--- 2. JAX (JIT + vmap possible, but loop first) ---")
    start = time.perf_counter()
    # Warmup + JIT compile
    _ = ElementStressRecoveryJax.recover_quad4_nodal(model, u_global_frames[0], sorted_nids, fields=fields)
    jax_time_start = time.perf_counter()
    for f in range(frames):
        res_jax = ElementStressRecoveryJax.recover_quad4_nodal(model, u_global_frames[f], sorted_nids, fields=fields)
    
    # wait for async dispatch
    jax.block_until_ready(res_jax["Stress"])
    jax_time = time.perf_counter() - jax_time_start
    print(f"JAX Time for {frames} frames: {jax_time:.4f}s ({frames/jax_time:.2f} FPS)")

    print("\n--- 3. Numba (@njit core) ---")
    start = time.perf_counter()
    # Warmup + JIT compile
    _ = ElementStressRecoveryNumba.recover_quad4_nodal(model, u_global_frames[0], sorted_nids, fields=fields)
    numba_time_start = time.perf_counter()
    for f in range(frames):
        res_nb = ElementStressRecoveryNumba.recover_quad4_nodal(model, u_global_frames[f], sorted_nids, fields=fields)
    numba_time = time.perf_counter() - numba_time_start
    print(f"Numba Time for {frames} frames: {numba_time:.4f}s ({frames/numba_time:.2f} FPS)")

    # Check accuracy
    print("\nChecking Accuracy (NumPy vs JAX / Numba)...")
    for k in res_np:
        # res_jax is jnp array, convert to np
        max_err_jax = np.max(np.abs(res_np[k] - np.array(res_jax[k])))
        max_err_nb = np.max(np.abs(res_np[k] - res_nb[k]))
        print(f"  Field {k}: Max Error (JAX) = {max_err_jax:.2e}, Max Error (Numba) = {max_err_nb:.2e}")
        if max_err_jax > 1e-4 or max_err_nb > 1e-4:
            print(f"  [!] WARNING: Large error in {k}")

if __name__ == "__main__":
    run_benchmark()
