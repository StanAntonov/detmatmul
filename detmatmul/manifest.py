"""
detmatmul.manifest
==================
Cross-hardware compliance manifest — build, save, load, compare.

The manifest is a JSON file that records the SHA-256 hashes produced by
this machine for a canonical set of test inputs. When run on a different
machine, the same hashes prove cross-hardware determinism.
"""

import os
import json
import platform
import hashlib
from datetime import datetime

import numpy as np

from detmatmul._version import __version__
from detmatmul.core import (
    SPEC_VERSION,
    SCALE_FACTOR,
    _matmul_raw,
)


# ── Default test suite ────────────────────────────────────────────────────────
def _default_cases():
    cases = []
    for seed in [42, 137, 999, 31415, 271828]:
        for (M, K, N) in [(64,64,64),(256,256,256),(512,128,512),
                          (128,512,128),(511,255,767)]:
            cases.append(dict(M=M, K=K, N=N, seed=seed, kind="normal", relu=True))
    cases.append(dict(M=64,  K=64,  N=64,  seed=0,  kind="zeros",     relu=True))
    cases.append(dict(M=64,  K=64,  N=64,  seed=0,  kind="ones",      relu=False))
    cases.append(dict(M=64,  K=64,  N=64,  seed=7,  kind="tiny",      relu=False))
    cases.append(dict(M=64,  K=32,  N=64,  seed=7,  kind="large",     relu=False))
    cases.append(dict(M=128, K=64,  N=128, seed=55, kind="neg_heavy", relu=True))
    cases.append(dict(M=300, K=150, N=200, seed=42, kind="odd_rect",  relu=False))
    return cases


def _make_matrices(case):
    rng  = np.random.default_rng(case["seed"])
    M, K, N, kind = case["M"], case["K"], case["N"], case["kind"]
    if kind == "zeros":
        return np.zeros((M,K), np.float32), np.zeros((K,N), np.float32)
    if kind == "ones":
        return np.ones((M,K), np.float32),  np.ones((K,N), np.float32)
    if kind == "tiny":
        return (rng.uniform(-1e-3,1e-3,(M,K)).astype(np.float32),
                rng.uniform(-1e-3,1e-3,(K,N)).astype(np.float32))
    if kind == "large":
        return (rng.uniform(-10,10,(M,K)).astype(np.float32),
                rng.uniform(-10,10,(K,N)).astype(np.float32))
    if kind == "neg_heavy":
        return (rng.uniform(-5,1,(M,K)).astype(np.float32),
                rng.uniform(-5,1,(K,N)).astype(np.float32))
    return (rng.standard_normal((M,K)).astype(np.float32),
            rng.standard_normal((K,N)).astype(np.float32))


def _case_key(case):
    return (f"{case['M']}x{case['K']}x{case['N']}_"
            f"{case['kind']}_seed{case['seed']}_"
            f"{'relu' if case['relu'] else 'norelu'}")


def _sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────
def build_manifest(
    force_cpu: bool = False,
    parallel_cpu: bool = False,
    gpu_name: str = "unknown",
    gpu_sm: str = "N/A",
) -> dict:
    """
    Run the canonical test suite and return a manifest dict.

    Parameters
    ----------
    force_cpu     : use CPU even if GPU is available
    parallel_cpu  : use multi-threaded CPU (still deterministic)
    gpu_name      : GPU name string for metadata
    gpu_sm        : compute capability string (e.g. "7.5")

    Returns
    -------
    dict with keys: spec_version, scale_factor, runs (list of 1)
    """
    cases = _default_cases()
    hashes = {}
    for case in cases:
        A, B    = _make_matrices(case)
        _, _, C = _matmul_raw(A, B, use_relu=case["relu"],
                               force_cpu=force_cpu,
                               parallel_cpu=parallel_cpu,
                               skip_overflow_check=True)
        hashes[_case_key(case)] = _sha256(C)

    run = {
        "run_id"    : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "gpu"       : gpu_name,
        "sm"        : gpu_sm,
        "host"      : platform.node(),
        "os"        : f"{platform.system()} {platform.release()}",
        "python"    : platform.python_version(),
        "detmatmul" : __version__,
        "timestamp" : datetime.now().isoformat(),
        "spec_ver"  : SPEC_VERSION,
        "mode"      : "cpu" if force_cpu else "gpu",
        "hashes"    : hashes,
    }
    return {
        "spec_version" : SPEC_VERSION,
        "scale_factor" : SCALE_FACTOR,
        "runs"         : [run],
    }


def load_manifest(path: str) -> dict:
    """
    Load a manifest from disk. Returns empty manifest if file doesn't exist
    or is corrupt/empty.
    """
    if os.path.exists(path):
        try:
            with open(path) as f:
                content = f.read().strip()
            if content:
                return json.loads(content)
        except (json.JSONDecodeError, OSError):
            pass
    return {"spec_version": SPEC_VERSION, "scale_factor": SCALE_FACTOR, "runs": []}


def save_manifest(manifest: dict, path: str) -> str:
    """
    Save manifest to disk. Returns the path actually written to.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def merge_manifests(manifests: list) -> dict:
    """
    Merge a list of manifest dicts, deduplicating by run_id.
    """
    merged   = {"spec_version": SPEC_VERSION, "scale_factor": SCALE_FACTOR, "runs": []}
    seen_ids = set()
    for m in manifests:
        for run in m.get("runs", []):
            uid = run.get("run_id", "") + "|" + run.get("gpu", "")
            if uid not in seen_ids:
                merged["runs"].append(run)
                seen_ids.add(uid)
    return merged


def compare_manifests(manifest: dict) -> dict:
    """
    Compare hashes across all runs in a manifest.

    Returns
    -------
    dict with keys:
      total    : int   — number of common test cases
      agreed   : int   — cases where all runs agree
      failed   : int   — cases where runs disagree
      skew     : int   — cases present in some runs but not all
      cases    : dict  — per-case results
      platforms: list  — distinct platform strings
      compliant: bool  — True if all common cases agree AND >1 distinct platform
    """
    runs = manifest.get("runs", [])
    if len(runs) < 2:
        return {"error": "Need at least 2 runs to compare.", "compliant": False}

    common_keys = set(runs[0]["hashes"].keys())
    for r in runs[1:]:
        common_keys.intersection_update(r["hashes"].keys())

    all_keys = set()
    for r in runs: all_keys.update(r["hashes"].keys())
    skew = len(all_keys) - len(common_keys)

    if not common_keys:
        return {
            "error": (
                "No common test cases. Each machine saved its own manifest. "
                "Use merge_manifests() to combine them first."
            ),
            "compliant": False,
        }

    results = {}
    for key in sorted(common_keys):
        run_data = [(r["gpu"], r["hashes"][key]) for r in runs]
        hashes   = [h for _, h in run_data]
        results[key] = {
            "agree"  : len(set(hashes)) == 1,
            "hashes" : run_data,
        }

    total     = len(results)
    agreed    = sum(1 for v in results.values() if v["agree"])
    platforms = list(set(f"{r['gpu']}|{r.get('os','?')}" for r in runs))
    compliant = (agreed == total) and len(platforms) > 1

    return {
        "total"    : total,
        "agreed"   : agreed,
        "failed"   : total - agreed,
        "skew"     : skew,
        "cases"    : results,
        "platforms": platforms,
        "compliant": compliant,
    }
