"""
tests/test_adapters.py
======================
Unit tests for JaxSSOAdapter and JaxFEMAdapter.

Uses lightweight mock objects instead of actual solver dependencies
so the tests run without JaxSSO / jax-fem installed.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wht_adapters import JaxSSOAdapter, JaxFEMAdapter
from wht_models import WHTMetadata, WHTResultData, WHTValidationError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_meta(analysis_type: str) -> WHTMetadata:
    return WHTMetadata(
        solver_name="JaxSSO", solver_version="0.1",
        analysis_type=analysis_type,
        coordinate_system="cartesian",
        unit_length="m", unit_force="N",
    )


def make_jaxsso_model(n_quads=2, n_beams=1):
    """Minimal mock JaxSSO model."""
    nodes = {i: [float(i), 0.0, 0.0] for i in range(n_quads * 4 + n_beams + 1)}
    quads = {}
    for i in range(n_quads):
        base = i * 4
        quads[i] = [base, base+1, base+2, base+3]
    beamcols = {n_quads: [0, 1]} if n_beams else {}
    model = SimpleNamespace(nodes=nodes, quads=quads,
                            beamcols=beamcols, truss={})
    return model


N_NODES = 9     # 2 quads (8 unique nodes) + 1 extra for beam end
N_MODES = 3
N_STEPS = 5


# ===========================================================================
# JaxSSOAdapter
# ===========================================================================

class TestJaxSSOAdapterStatic:

    def test_basic_conversion(self):
        model  = make_jaxsso_model()
        N      = len(model.nodes)
        u      = np.random.randn(N, 3)
        meta   = make_meta("static")
        data   = JaxSSOAdapter().convert(model, {"u": u}, "static", meta)

        assert isinstance(data, WHTResultData)
        assert data.n_nodes    == N
        assert data.n_timesteps == 1
        assert "Displacement" in data.point_data
        assert data.point_data["Displacement"].shape == (1, N, 3)

    def test_missing_key_raises(self):
        with pytest.raises(WHTValidationError, match="missing required keys"):
            JaxSSOAdapter().convert(
                make_jaxsso_model(), {}, "static", make_meta("static")
            )

    def test_unsupported_analysis_raises(self):
        with pytest.raises(WHTValidationError, match="does not support"):
            JaxSSOAdapter().convert(
                make_jaxsso_model(), {}, "transient", make_meta("static")
            )


class TestJaxSSOAdapterModal:

    def test_basic_conversion(self):
        model  = make_jaxsso_model()
        N      = len(model.nodes)
        vecs   = np.random.randn(N_MODES, N, 3)
        freqs  = np.array([1.0, 2.5, 4.8])
        meta   = make_meta("modal")
        data   = JaxSSOAdapter().convert(
            model, {"vecs": vecs, "freqs": freqs}, "modal", meta
        )

        assert data.n_timesteps == N_MODES
        assert data.point_data["Displacement"].shape == (N_MODES, N, 3)
        np.testing.assert_array_equal(data.time_values, freqs)

    def test_negative_freq_raises(self):
        model  = make_jaxsso_model()
        N      = len(model.nodes)
        meta   = make_meta("modal")
        with pytest.raises(WHTValidationError, match="positive"):
            JaxSSOAdapter().convert(
                model,
                {"vecs": np.ones((2, N, 3)), "freqs": np.array([-1.0, 2.0])},
                "modal", meta,
            )


class TestJaxSSOAdapterBuckling:

    def test_basic_conversion(self):
        model        = make_jaxsso_model()
        N            = len(model.nodes)
        modes        = np.random.randn(N_MODES, N, 3)
        load_factors = np.array([1.2, 3.5, 7.1])
        meta         = make_meta("buckling")
        data         = JaxSSOAdapter().convert(
            model,
            {"modes": modes, "load_factors": load_factors},
            "buckling", meta,
        )

        assert data.n_timesteps == N_MODES
        assert "BucklingMode" in data.point_data
        assert "LoadFactor"   in data.field_data
        np.testing.assert_array_equal(data.time_values, load_factors)

    def test_negative_load_factor_warns(self):
        model = make_jaxsso_model()
        N     = len(model.nodes)
        meta  = make_meta("buckling")
        with pytest.warns(match="non-physical"):
            JaxSSOAdapter().convert(
                model,
                {"modes": np.ones((2, N, 3)), "load_factors": np.array([-1.0, 2.0])},
                "buckling", meta,
            )

    def test_empty_element_groups_raises(self):
        model = SimpleNamespace(
            nodes={0: [0, 0, 0], 1: [1, 0, 0]},
            quads={}, beamcols={}, truss={},
        )
        with pytest.raises(WHTValidationError, match="no recognized element"):
            JaxSSOAdapter().convert(
                model, {"modes": np.ones((1, 2, 3)),
                        "load_factors": np.array([1.0])},
                "buckling", make_meta("buckling"),
            )


# ===========================================================================
# JaxFEMAdapter
# ===========================================================================

def make_jaxfem_problem(N=6, n_elems=2, nodes_per_elem=3):
    """Mock jax-fem problem with mesh.node_coords and mesh.cells."""
    coords = np.random.randn(N, 3)
    cells  = np.array([[i, i+1, i+2] for i in range(n_elems)], dtype=np.int64)
    mesh   = SimpleNamespace(node_coords=coords, cells=cells)
    return SimpleNamespace(mesh=mesh)


class TestJaxFEMAdapterStatic:

    def test_basic_conversion(self):
        prob = make_jaxfem_problem(N=6)
        N    = prob.mesh.node_coords.shape[0]
        u    = np.random.randn(N, 3)
        meta = WHTMetadata("jax-fem", "0.1", "static", "cartesian", "m", "N")
        data = JaxFEMAdapter().convert(prob, {"u": u}, "static", meta)

        assert data.n_nodes    == N
        assert data.n_timesteps == 1

    def test_2d_nodes_padded(self):
        """2D node_coords (N, 2) should be padded with Z=0."""
        coords = np.random.randn(4, 2)
        cells  = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        mesh   = SimpleNamespace(node_coords=coords, cells=cells)
        prob   = SimpleNamespace(mesh=mesh)
        u      = np.random.randn(4, 2)
        meta   = WHTMetadata("jax-fem", "0.1", "static", "cartesian", "mm", "N")
        data   = JaxFEMAdapter().convert(prob, {"u": u}, "static", meta)
        assert data.nodes.shape == (4, 3)


class TestJaxFEMAdapterTransient:

    def test_basic_conversion(self):
        prob   = make_jaxfem_problem(N=6)
        N      = prob.mesh.node_coords.shape[0]
        T      = N_STEPS
        u      = np.random.randn(T, N, 3)
        t      = np.linspace(0, 1, T)
        meta   = WHTMetadata("jax-fem", "0.1", "transient", "cartesian", "m", "N")
        data   = JaxFEMAdapter().convert(prob, {"u": u, "t": t}, "transient", meta)

        assert data.n_timesteps == T
        np.testing.assert_array_almost_equal(data.time_values, t)

    def test_non_monotonic_time_raises(self):
        prob = make_jaxfem_problem(N=6)
        N    = prob.mesh.node_coords.shape[0]
        meta = WHTMetadata("jax-fem", "0.1", "transient", "cartesian", "m", "N")
        with pytest.raises(WHTValidationError, match="monotonically"):
            JaxFEMAdapter().convert(
                prob,
                {"u": np.ones((3, N, 3)), "t": np.array([0.0, 0.5, 0.3])},
                "transient", meta,
            )


class TestJaxFEMAdapterModal:

    def test_basic_conversion(self):
        prob    = make_jaxfem_problem(N=6)
        N       = prob.mesh.node_coords.shape[0]
        eigvecs = np.random.randn(N_MODES, N, 3)
        # rad²/s² values
        eigvals = np.array([(2 * np.pi * f)**2 for f in [1.0, 2.0, 3.0]])
        meta    = WHTMetadata("jax-fem", "0.1", "modal", "cartesian", "m", "N")
        data    = JaxFEMAdapter().convert(
            prob, {"eigvecs": eigvecs, "eigvals": eigvals}, "modal", meta
        )

        assert data.n_timesteps == N_MODES
        # Frequencies should be approximately [1, 2, 3] Hz
        np.testing.assert_allclose(data.time_values, [1.0, 2.0, 3.0], atol=1e-6)
