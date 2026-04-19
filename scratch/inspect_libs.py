import sys
import numpy as np
try:
    from JaxSSO.model import Model
    model = Model()
    print("JaxSSO Model attributes:", dir(model))
    # Test node/element storage
    model.add_node(1, 0.0, 0.0, 0.0)
    # Check if we can find where nodes are stored
    # Based on typical JaxSSO, it might be in model.nodes or similar
except Exception as e:
    print(f"JaxSSO inspection failed: {e}")

try:
    import jax_fem
    print("jax-fem attributes:", dir(jax_fem))
    # Typically jax-fem uses problem.mesh
except Exception as e:
    print(f"jax-fem inspection failed: {e}")
