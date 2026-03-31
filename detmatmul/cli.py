"""
detmatmul.cli
=============
Command-line entry points registered in pyproject.toml.

    detmatmul-audit   — run the full 31-case audit on this machine
    detmatmul-verify  — verify this machine against a saved manifest
    detmatmul-merge   — merge manifests from multiple machines
"""

import argparse
import os
import sys
import platform

from detmatmul._version import __version__
from detmatmul.core import SPEC_VERSION, SCALE_FACTOR
from detmatmul.manifest import (
    build_manifest,
    load_manifest,
    save_manifest,
    merge_manifests,
    compare_manifests,
)

# ── GPU name detection (optional) ────────────────────────────────────────────
_GPU_NAME = "CPU-only"
_GPU_SM   = "N/A"
try:
    from numba import cuda as _cuda
    _dev      = _cuda.get_current_device()
    _GPU_NAME = _dev.name.decode() if isinstance(_dev.name, bytes) else _dev.name
    _cc       = _dev.compute_capability
    _GPU_SM   = f"{_cc[0]}.{_cc[1]}"
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _print_comparison(manifest: dict) -> bool:
    """Print cross-hardware proof table. Returns True if compliant."""
    result = compare_manifests(manifest)

    if "error" in result:
        print(f"\n  {result['error']}")
        return False

    runs      = manifest.get("runs", [])
    platforms = list({f"{r['gpu']} | {r.get('os','?')}" for r in runs})
    agreed    = result["agreed"]
    total     = result["total"]
    failed    = result["failed"]
    compliant = result["compliant"]

    w = 68
    print(f"\n  {'='*w}")
    print(f"  Cross-Hardware Compliance Report")
    print(f"  {'='*w}")
    print(f"\n  Platforms compared ({len(platforms)}):")
    for p in platforms:
        print(f"    • {p}")
    print(f"\n  Results: {agreed}/{total} cases agree", end="")
    if failed:
        print(f"  —  {failed} MISMATCH(ES)")
        for key, v in result["cases"].items():
            if not v["agree"]:
                print(f"\n    FAIL  {key}")
                for gpu, h in v["hashes"]:
                    print(f"      {gpu[:36]}: {h[:32]}...")
    else:
        print("  —  zero mismatches")

    print(f"\n  {'='*w}")
    if compliant:
        print("  PROOF COMPLETE")
        print(f"  {len(platforms)} distinct platforms. {total} hashes. Zero mismatches.")
        print("  Q16.16 fixed-point integer arithmetic is cross-hardware deterministic.")
    elif agreed == total:
        print("  Hashes agree but only one distinct platform — run on different hardware")
        print("  to generate a cross-hardware proof.")
    else:
        print(f"  NON-COMPLIANT: {failed} mismatch(es) detected.")
    print(f"  {'='*w}\n")
    return compliant


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def main_audit(argv=None):
    """
    Run the full 31-case benchmark on this machine and save a manifest.

    Usage:
        detmatmul-audit
        detmatmul-audit --cpu
        detmatmul-audit --parallel-cpu
        detmatmul-audit --output my_manifest.json
    """
    parser = argparse.ArgumentParser(
        prog="detmatmul-audit",
        description="Run the DIS v1.0 canonical test suite and save a manifest.",
    )
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode (no CUDA required)")
    parser.add_argument("--parallel-cpu", action="store_true",
                        help="Use multi-threaded CPU kernel (faster, still deterministic)")
    parser.add_argument("--output", default="hash_manifest.json",
                        metavar="FILE",
                        help="Path to save the manifest (default: hash_manifest.json)")
    parser.add_argument("--version", action="store_true",
                        help="Print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(f"detmatmul {__version__}  /  DIS spec v{SPEC_VERSION}")
        sys.exit(0)

    force_cpu    = args.cpu
    parallel_cpu = args.parallel_cpu
    gpu_name     = "CPU-only" if force_cpu else _GPU_NAME
    gpu_sm       = "N/A"      if force_cpu else _GPU_SM
    mode         = "cpu"      if (force_cpu or _GPU_NAME == "CPU-only") else "gpu"

    print(f"\ndetmatmul {__version__}  —  DIS v{SPEC_VERSION}  (Q16.16, scale={SCALE_FACTOR})")
    print(f"Hardware : {gpu_name}")
    print(f"OS       : {platform.system()} {platform.release()}")
    print(f"Mode     : {mode}" + (" (parallel)" if parallel_cpu else ""))
    print()
    print("Running 31 canonical test cases...")

    manifest = build_manifest(
        force_cpu    = force_cpu,
        parallel_cpu = parallel_cpu,
        gpu_name     = gpu_name,
        gpu_sm       = gpu_sm,
    )

    n = len(manifest["runs"][0]["hashes"])
    print(f"  {n}/31 cases completed.\n")

    # Print sample hashes
    hashes = manifest["runs"][0]["hashes"]
    print(f"  {'Test case':<45}  SHA-256 (first 32 chars)")
    print(f"  {'-'*68}")
    for k in list(hashes)[:6]:
        print(f"  {k:<45}  {hashes[k][:32]}...")
    print(f"  ... and {n-6} more\n")

    # Append to existing manifest if present
    existing = load_manifest(args.output)
    existing.setdefault("spec_version", SPEC_VERSION)
    existing.setdefault("scale_factor", SCALE_FACTOR)
    existing.setdefault("runs", [])

    new_run = manifest["runs"][0]
    existing["runs"].append(new_run)

    save_manifest(existing, args.output)
    print(f"Manifest saved → {os.path.abspath(args.output)}")
    print(f"({len(existing['runs'])} total run(s) in file)\n")

    if len(existing["runs"]) >= 2:
        _print_comparison(existing)
    else:
        print("This is the first entry. To generate a cross-hardware proof:")
        print("  1. Run this command on a different machine")
        print("  2. Merge the two manifests:")
        print(f"     detmatmul-merge hash_manifest.json other_manifest.json\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

def main_verify(argv=None):
    """
    Verify this machine's hashes against a reference manifest.

    Usage:
        detmatmul-verify manifests/hash_manifest.json
        detmatmul-verify manifests/hash_manifest.json --cpu
    """
    parser = argparse.ArgumentParser(
        prog="detmatmul-verify",
        description="Verify this machine against a reference manifest.",
    )
    parser.add_argument("manifest",
                        help="Path to reference manifest JSON")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--parallel-cpu", action="store_true",
                        help="Use parallel CPU kernel")
    args = parser.parse_args(argv)

    ref = load_manifest(args.manifest)
    if not ref.get("runs"):
        print(f"ERROR: No runs found in {args.manifest}")
        sys.exit(1)

    force_cpu = args.cpu
    gpu_name  = "CPU-only" if force_cpu else _GPU_NAME

    print(f"\ndetmatmul {__version__}  —  Verify mode")
    print(f"Reference : {args.manifest}  ({len(ref['runs'])} run(s))")
    print(f"This      : {gpu_name}  |  {platform.system()} {platform.release()}")
    print()

    local = build_manifest(
        force_cpu    = force_cpu,
        parallel_cpu = args.parallel_cpu,
        gpu_name     = gpu_name,
        gpu_sm       = _GPU_SM,
    )

    # Compare local run against each reference run
    local_hashes = local["runs"][0]["hashes"]
    all_ok       = True

    for ref_run in ref["runs"]:
        ref_hashes = ref_run["hashes"]
        common     = set(local_hashes) & set(ref_hashes)
        mismatches = [k for k in common if local_hashes[k] != ref_hashes[k]]

        status = "OK" if not mismatches else "FAIL"
        print(f"  [{status}]  vs {ref_run['gpu']} ({ref_run.get('os','?')})")
        if mismatches:
            all_ok = False
            for k in mismatches:
                print(f"    MISMATCH  {k}")
                print(f"      ref : {ref_hashes[k][:40]}...")
                print(f"      got : {local_hashes[k][:40]}...")

    print()
    if all_ok:
        print("COMPLIANT — all hashes match the reference manifest.")

        # Save updated manifest with this run appended
        combined = merge_manifests([ref, local])
        out_path = "verified_manifest.json"
        save_manifest(combined, out_path)
        print(f"Combined manifest saved → {os.path.abspath(out_path)}")
    else:
        print("NON-COMPLIANT — hash mismatch(es) detected.")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE
# ═══════════════════════════════════════════════════════════════════════════════

def main_merge(argv=None):
    """
    Merge manifests from multiple machines and print the compliance proof.

    Usage:
        detmatmul-merge machine_a.json machine_b.json
        detmatmul-merge *.json --output merged.json
    """
    parser = argparse.ArgumentParser(
        prog="detmatmul-merge",
        description="Merge manifests from multiple machines.",
    )
    parser.add_argument("manifests", nargs="+",
                        help="Manifest JSON files to merge")
    parser.add_argument("--output", default="merged_manifest.json",
                        metavar="FILE",
                        help="Output path (default: merged_manifest.json)")
    args = parser.parse_args(argv)

    loaded = []
    for path in args.manifests:
        if not os.path.exists(path):
            print(f"WARNING: not found — {path}")
            continue
        m = load_manifest(path)
        loaded.append(m)
        print(f"  Loaded {path}  ({len(m.get('runs', []))} run(s))")

    if not loaded:
        print("ERROR: no valid manifests found.")
        sys.exit(1)

    merged   = merge_manifests(loaded)
    n_runs   = len(merged["runs"])
    n_plat   = len({f"{r['gpu']}|{r.get('os','')}" for r in merged["runs"]})

    print(f"\nMerged: {n_runs} total run(s) across {n_plat} distinct platform(s)")

    save_manifest(merged, args.output)
    print(f"Saved  → {os.path.abspath(args.output)}\n")

    if n_runs >= 2:
        _print_comparison(merged)
    else:
        print("Only 1 run after merge — need at least 2 to compare.")
