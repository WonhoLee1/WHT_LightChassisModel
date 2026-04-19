"""
__main__.py
===========
WHT Universal FEM Result Converter — CLI Entry Point

Usage
-----
    python -m wht_converter --help

    # Export all formats
    python -m wht_converter --solver jaxsso --analysis modal \
        --input exam1_nf.py --export all --output results/

    # VTKHDF only, with compression
    python -m wht_converter --solver jaxsso --analysis modal \
        --input exam1_nf.py --export vtkhdf \
        --compression gzip --compression-level 6 --output results/

    # Dry-run: validate IR without writing files
    python -m wht_converter --solver jaxsso --analysis modal \
        --input exam1_nf.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

from .wht_exporters import HWASCIIExporter, VTKHDFExporter, VTUPVDExporter
from .wht_models import WHTExportWarning, WHTValidationError


# ---------------------------------------------------------------------------
# Exporter registry
# ---------------------------------------------------------------------------

EXPORTER_MAP = {
    "vtkhdf":  VTKHDFExporter,
    "vtu":     VTUPVDExporter,
    "hwascii": HWASCIIExporter,
}

OUTPUT_EXT = {
    "vtkhdf":  ".hdf",
    "vtu":     ".pvd",
    "hwascii": ".ascii",
}


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wht_converter",
        description="WHT Universal FEM Result Converter v0.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to solver script that exposes get_wht_data() → WHTResultData.",
    )
    parser.add_argument(
        "--export", "-e",
        choices=["all", "vtkhdf", "vtu", "hwascii"],
        default="all",
        help="Output format(s). 'all' writes vtkhdf + vtu + hwascii. (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/",
        help="Output directory. Files are named after --input stem. (default: results/)",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="HDF5 compression for VTKHDF output. (default: gzip)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=4,
        metavar="LEVEL",
        help="gzip compression level 1–9. (default: 4)",
    )
    parser.add_argument(
        "--chunk-timesteps", type=int, default=10,
        metavar="N",
        help="Timesteps per HDF5 chunk. (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and display IR summary without writing files.",
    )
    parser.add_argument(
        "--warn-errors", action="store_true",
        help="Treat WHTExportWarning as errors (strict mode).",
    )
    return parser


# ---------------------------------------------------------------------------
# Script loader
# ---------------------------------------------------------------------------

def load_wht_data_from_script(script_path: str):
    """
    Dynamically import a solver script and call its ``get_wht_data()``
    function to obtain a ``WHTResultData`` object.

    The solver script must define:

        def get_wht_data() -> WHTResultData:
            ...

    Example (exam1_nf.py)
    ---------------------
        from wht_adapters import JaxSSOAdapter
        from wht_models import WHTMetadata

        def get_wht_data():
            model  = ...  # run JaxSSO
            freqs  = ...
            vecs   = ...
            meta   = WHTMetadata(solver_name="JaxSSO", ...)
            adapter = JaxSSOAdapter()
            return adapter.convert(model, {"vecs": vecs, "freqs": freqs},
                                   "modal", meta)
    """
    path = Path(script_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input script not found: {path}")

    spec   = importlib.util.spec_from_file_location("_wht_input_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_wht_data"):
        raise AttributeError(
            f"Script '{path.name}' must define a function: "
            f"get_wht_data() -> WHTResultData"
        )
    return module.get_wht_data()


# ---------------------------------------------------------------------------
# Export dispatch
# ---------------------------------------------------------------------------

def run_export(
    data,
    export_key: str,
    output_dir: str,
    stem: str,
    compression: str,
    compression_level: int,
    chunk_timesteps: int,
    dry_run: bool,
) -> None:
    keys = list(EXPORTER_MAP.keys()) if export_key == "all" else [export_key]

    for key in keys:
        ext      = OUTPUT_EXT[key]
        out_path = str(Path(output_dir) / f"{stem}{ext}")

        if dry_run:
            print(f"  [dry-run] Would write: {out_path}")
            continue

        if key == "vtkhdf":
            comp = None if compression == "none" else compression
            exporter = VTKHDFExporter(
                compression=comp,
                compression_opts=compression_level,
                chunk_timesteps=chunk_timesteps,
            )
        elif key == "vtu":
            exporter = VTUPVDExporter()
        else:
            exporter = HWASCIIExporter()

        exporter.export(data, out_path)
        print(f"  ✓ {key.upper():8s} → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.warn_errors:
        warnings.filterwarnings("error", category=WHTExportWarning)

    print(f"\nWHT Converter v0.4")
    print(f"  Input  : {args.input}")
    print(f"  Export : {args.export}")
    print(f"  Output : {args.output}")
    print()

    # 1. Load data from solver script
    try:
        print("Loading WHTResultData from solver script …")
        data = load_wht_data_from_script(args.input)
        print(f"  {data}")
    except (FileNotFoundError, AttributeError, WHTValidationError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    # 2. Summary / dry-run
    print(f"\n  Nodes    : {data.n_nodes}")
    print(f"  Elements : {data.n_cells}")
    print(f"  Steps    : {data.n_timesteps}")
    print(f"  Analysis : {data.metadata.analysis_type}")
    print(f"  PointData: {list(data.point_data.keys())}")
    print(f"  CellData : {list(data.cell_data.keys())}")

    if args.dry_run:
        print("\n[dry-run] IR is valid. No files written.")
        return 0

    # 3. Export
    print(f"\nExporting …")
    stem = Path(args.input).stem
    try:
        run_export(
            data=data,
            export_key=args.export,
            output_dir=args.output,
            stem=stem,
            compression=args.compression,
            compression_level=args.compression_level,
            chunk_timesteps=args.chunk_timesteps,
            dry_run=args.dry_run,
        )
    except WHTValidationError as exc:
        print(f"\n[VALIDATION ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[EXPORT ERROR] {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
