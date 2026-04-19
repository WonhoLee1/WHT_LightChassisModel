"""
wht_converter
=============
WHT Universal FEM Result Converter v0.4

A clean, decoupled pipeline between JAX-based FEM solvers
(JaxSSO, jax-fem) and industrial post-processing tools
(ParaView, Altair HyperView).

Quick start
-----------
    from wht_converter.wht_models   import WHTMetadata, WHTResultData
    from wht_converter.wht_adapters import JaxSSOAdapter, JaxFEMAdapter
    from wht_converter.wht_exporters import VTKHDFExporter, HWASCIIExporter

    meta    = WHTMetadata(solver_name="JaxSSO", solver_version="0.1.0",
                          analysis_type="modal", coordinate_system="cartesian",
                          unit_length="m", unit_force="N")
    adapter  = JaxSSOAdapter()
    data     = adapter.convert(model, {"vecs": vecs, "freqs": freqs},
                               "modal", meta)

    VTKHDFExporter().export(data, "results/output.hdf")
    HWASCIIExporter().export(data, "results/output.ascii")
"""

from .wht_models import (
    WHTMetadata,
    WHTResultData,
    WHTValidationError,
    WHTExportWarning,
)
from .wht_utils import (
    VTKCellType,
    to_vtk_csr,
    merge_csr,
    node_dict_to_array,
    remap_connectivity,
)
from .wht_adapters import (
    BaseAdapter,
    JaxSSOAdapter,
    JaxFEMAdapter,
)
from .wht_exporters import (
    BaseExporter,
    VTKHDFExporter,
    VTUPVDExporter,
    HWASCIIExporter,
)

__version__ = "0.4.0"
__all__ = [
    "WHTMetadata", "WHTResultData", "WHTValidationError", "WHTExportWarning",
    "VTKCellType", "to_vtk_csr", "merge_csr",
    "node_dict_to_array", "remap_connectivity",
    "BaseAdapter", "JaxSSOAdapter", "JaxFEMAdapter",
    "BaseExporter", "VTKHDFExporter", "VTUPVDExporter", "HWASCIIExporter",
]
