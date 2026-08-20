#!/usr/bin/env python3
"""
cp4_to_cp2.py — Convert Cellpose 4 _seg.npy files to Cellpose 2-compatible format.

Always re-saves every file (even seemingly valid ones) to ensure the .npy
format is written by the current numpy environment, fixing pickle/format
incompatibilities between CP4 (newer numpy) and CP2 (older numpy).
Only ADDS missing/None keys — never modifies existing data values.

Usage:
    python cp4_to_cp2.py image_seg.npy
    python cp4_to_cp2.py image_seg.npy -o output_seg.npy
    python cp4_to_cp2.py --batch /path/to/directory/
    python cp4_to_cp2.py --batch /path/to/directory/ --suffix _cp2
    python cp4_to_cp2.py image_seg.npy --dry-run -v
"""

import argparse
import io
import pickle
import sys
from pathlib import Path


_NPY_MAGIC = b"\x93NUMPY"
_COMPAT_PICKLER_BASE = getattr(pickle, "_Pickler", pickle.Pickler)


class _Compat1xPickler(_COMPAT_PICKLER_BASE):
    """Rewrite NumPy 2.x private module paths for NumPy 1.x loaders."""

    def save_global(self, obj, name=None):
        if name is None:
            name = getattr(obj, "__qualname__", None) or obj.__name__
        module = pickle.whichmodule(obj, name)
        if module and "numpy._core" in module:
            compat_module = module.replace("numpy._core", "numpy.core")
            self.write(
                b"c"
                + compat_module.encode("utf-8") + b"\n"
                + name.encode("utf-8") + b"\n"
            )
            self.memoize(obj)
            return
        super().save_global(obj, name)


def make_colors(n: int):
    import numpy as np
    rng = np.random.default_rng(seed=42)
    return rng.integers(50, 230, size=(max(n, 1), 3), dtype=np.int32)


def _load_seg_payload(input_path: Path):
    """Load either a proper .npy file or a plain pickle disguised as .npy."""
    import numpy as np

    with input_path.open("rb") as handle:
        header = handle.read(len(_NPY_MAGIC))

    if header == _NPY_MAGIC:
        raw = np.load(input_path, allow_pickle=True)
    else:
        with input_path.open("rb") as handle:
            raw = pickle.load(handle)

    if isinstance(raw, np.ndarray):
        return raw.item()

    return raw


def _save_compat_npy(output_path: Path, data) -> None:
    """Write a real .npy container whose pickle payload loads on NumPy 1.x."""
    import numpy as np

    payload = np.asarray(data, dtype=object)

    header_buffer = io.BytesIO()
    np.save(header_buffer, payload)
    header_bytes = header_buffer.getvalue()

    header_buffer.seek(6)
    major = header_buffer.read(1)[0]
    header_buffer.read(1)
    header_len = (
        int.from_bytes(header_buffer.read(2), "little") if major == 1
        else int.from_bytes(header_buffer.read(4), "little")
    )
    header_buffer.read(header_len)
    header_end = header_buffer.tell()

    pickle_buffer = io.BytesIO()
    _Compat1xPickler(pickle_buffer, protocol=3).dump(payload)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(header_bytes[:header_end])
        handle.write(pickle_buffer.getvalue())
    tmp_path.replace(output_path)


def _audit_saved_file(output_path: Path) -> list[str]:
    """Return compatibility problems that would still trip Cellpose 2."""
    import numpy as np

    problems: list[str] = []
    payload = output_path.read_bytes()
    if payload[:len(_NPY_MAGIC)] != _NPY_MAGIC:
        problems.append("output is not a real .npy file")

    if b"numpy._core" in payload:
        problems.append("output still contains numpy._core references")

    try:
        dat = np.load(output_path, allow_pickle=True).item()
    except Exception as exc:
        problems.append(f"output failed np.load(...).item(): {exc}")
        return problems

    if not isinstance(dat, dict):
        problems.append(f"output is not a dict after load (got {type(dat).__name__})")
        return problems

    if "outlines" not in dat:
        problems.append("output is missing 'outlines'")

    return problems


def patch_and_save(input_path: Path, output_path: Path, verbose: bool = False) -> str:
    """Load, patch missing keys, force re-save in current numpy format.
    Re-saving rewrites the pickle with the current numpy env, which fixes
    CP4→CP2 format incompatibilities.
    Returns: 'patched' or 'error'
    """
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
        sys.exit(1)

    try:
        dat = _load_seg_payload(input_path)
    except Exception as e:
        print(f"  ERROR loading: {e}", file=sys.stderr)
        return "error"

    if not isinstance(dat, dict):
        print(f"  ERROR: not a dict (got {type(dat).__name__})", file=sys.stderr)
        return "error"

    if "masks" not in dat:
        print(f"  ERROR: 'masks' key missing. Keys: {list(dat.keys())}", file=sys.stderr)
        return "error"

    masks = dat["masks"]
    n_masks = int(masks.max())

    if verbose:
        print(f"  Keys in file  : {list(dat.keys())}")

    added = []

    # Only insert keys that are absent or None; never modify existing values
    if "outlines" not in dat or dat["outlines"] is None:
        dat["outlines"] = np.zeros_like(masks, dtype=masks.dtype)
        added.append("outlines")

    if "colors" not in dat or dat["colors"] is None:
        dat["colors"] = make_colors(n_masks)
        added.append("colors")

    if "chan_choose" not in dat or dat["chan_choose"] is None:
        dat["chan_choose"] = [0, 0]
        added.append("chan_choose")

    if "img" not in dat or dat["img"] is None:
        dat["img"] = np.zeros(masks.shape, dtype=np.float32)
        added.append("img (blank — original image not in CP4 file)")
        if verbose:
            print("  WARNING: 'img' missing; CP2 GUI will load but won't display image")

    if "ismanual" not in dat or dat["ismanual"] is None:
        dat["ismanual"] = np.zeros(n_masks, dtype=bool)
        added.append("ismanual")

    if "flows" not in dat or dat["flows"] is None:
        dat["flows"] = [[], [], [], [], [[]]]
        added.append("flows")

    if "manual_changes" not in dat:
        dat["manual_changes"] = []
        added.append("manual_changes")

    if "model_path" not in dat:
        dat["model_path"] = 0
        added.append("model_path")

    if "flow_threshold" not in dat or dat["flow_threshold"] is None:
        dat["flow_threshold"] = 0.4
        added.append("flow_threshold")

    if "cellprob_threshold" not in dat or dat["cellprob_threshold"] is None:
        dat["cellprob_threshold"] = 0.0
        added.append("cellprob_threshold")

    if verbose:
        if added:
            print(f"  Added keys    : {added}")
        else:
            print("  Keys complete — re-saving for numpy compatibility only")

    # Always force a full re-save into a real .npy container with a NumPy 1.x
    # compatible pickle payload.
    try:
        _save_compat_npy(output_path, dat)
    except Exception as e:
        print(f"  ERROR saving: {e}", file=sys.stderr)
        return "error"

    problems = _audit_saved_file(output_path)
    if problems:
        print(f"  ERROR: compatibility audit failed: {problems}", file=sys.stderr)
        return "error"

    if verbose:
        print(f"  Saved → {output_path}")

    return "patched"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cellpose 4 _seg.npy files to Cellpose 2-compatible format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy: always re-saves (even if all keys present) to fix numpy version
incompatibilities. Only adds missing/None keys; never modifies existing data.

Examples:
  python cp4_to_cp2.py image_seg.npy
  python cp4_to_cp2.py image_seg.npy -o image_cp2_seg.npy
  python cp4_to_cp2.py img1_seg.npy img2_seg.npy img3_seg.npy
  python cp4_to_cp2.py --batch /path/to/directory/
  python cp4_to_cp2.py --batch /path/to/directory/ --suffix _cp2
  python cp4_to_cp2.py image_seg.npy --dry-run -v
        """,
    )
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Input _seg.npy file(s) to convert.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path. Single-file only. Omit to overwrite in-place.")
    parser.add_argument("--batch", action="store_true",
                        help="Scan a directory for all *_seg.npy files.")
    parser.add_argument("--suffix", type=str, default=None, metavar="SUFFIX",
                        help="Save alongside original with a name suffix "
                             "(e.g. --suffix _cp2 → image_seg_cp2.npy). "
                             "Safe: won't overwrite the original.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-file details.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without saving.")

    args = parser.parse_args()

    files: list[Path] = []
    if args.batch:
        dirs = [p for p in args.inputs if p.is_dir()] if args.inputs else [Path(".")]
        for d in dirs:
            found = sorted(d.glob("*_seg.npy"))
            if not found:
                print(f"  No *_seg.npy files found in {d}", file=sys.stderr)
            files.extend(found)
        files.extend(p for p in args.inputs if p.is_file())
    else:
        if not args.inputs:
            parser.print_help()
            sys.exit(0)
        files = list(args.inputs)

    if not files:
        print("No input files.", file=sys.stderr)
        sys.exit(1)

    if args.output and len(files) > 1:
        print("ERROR: -o can only be used with a single file.", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for src in files:
        if not src.exists():
            print(f"ERROR: not found: {src}", file=sys.stderr)
            fail += 1
            continue

        dst = args.output if args.output else (
            src.with_name(src.stem + args.suffix + ".npy") if args.suffix else src
        )

        if args.dry_run:
            print(f"[dry-run] {src.name}")
            try:
                dat = _load_seg_payload(src)
                missing = [k for k in [
                    "outlines", "colors", "chan_choose", "img", "ismanual",
                    "flows", "manual_changes", "model_path",
                    "flow_threshold", "cellprob_threshold"]
                    if k not in dat or dat[k] is None]
                print(f"  Would add : {missing if missing else '(none)'}")
                print(f"  Would save: {dst}")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
            continue

        print(f"  {src.name}", end=" → ")
        result = patch_and_save(src, dst, verbose=args.verbose)
        print(result)
        if result == "patched":
            ok += 1
        else:
            fail += 1

    if len(files) > 1 or fail > 0:
        print(f"\nDone: {ok} converted, {fail} failed.")

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
