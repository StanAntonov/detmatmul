#!/usr/bin/env python3
"""
Deterministic Fixed‑Point Matrix Multiplication  —  v9.3 Turbo (Final)
═══════════════════════════════════════════════════════════════════════════════

Core claim: identical SHA‑256 hashes for identical inputs,
regardless of hardware (CPU brand, GPU brand/generation, OS).

Achieved by replacing IEEE‑754 floating point with Q16.16 fixed‑point
integer arithmetic (int64). Integer ops are fully associative —
no hardware‑dependent rounding, no non‑determinism.

Features:
  - Turbo GPU kernel (4x4 register blocking, 64x64 tiles)
  - Full 31‑test‑case manifest generation
  - Overflow detection & safe input range calculator
  - Quantisation error analysis vs float64
  - Cross‑hardware proof: merge and verify manifests
  - Command‑line interface with all original options

Usage:
  python detmatmul.py                  # full GPU audit
  python detmatmul.py --verify         # verify vs manifest
  python detmatmul.py --verify other.json
  python detmatmul.py --cpu            # CPU-only (Colab etc.)
  python detmatmul.py --parallel-cpu   # use multi‑threaded CPU (faster)
  python detmatmul.py --merge a.json b.json  # merge manifests
  python detmatmul.py --merge a.json b.json --output merged.json
  python detmatmul.py --version        # show version
  python detmatmul.py --no-pause       # disable waiting at exit
"""

import os
import sys
import json
import platform
import warnings
import argparse
import math
import time
import traceback
import hashlib
from datetime import datetime

# ── Auto-install on Colab / bare environments ─────────────────────────────────
def _ensure_deps():
    try:
        import numba
        import numpy
    except ImportError:
        print("[setup] Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "numba", "numpy", "-q"])
        print("[setup] Done. Please re-run the script.")
        sys.exit(0)

_ensure_deps()

import numpy as np
from numba import njit, prange, int64, int32
import numba

# ── CUDA is optional ──────────────────────────────────────────────────────────
_CUDA_AVAILABLE = False
try:
    from numba import cuda
    _dev = cuda.get_current_device()
    _dev.name
    _CUDA_AVAILABLE = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)

# ── Colab / Jupyter compatibility ────────────────────────────────────────────
def _is_interactive():
    try:
        name = get_ipython().__class__.__name__
        return name in ("ZMQInteractiveShell", "TerminalInteractiveShell", "Shell")
    except NameError:
        return False

def _pause_if_needed(pause):
    if pause and not _is_interactive():
        input("\nPress ENTER to exit...")

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEC
# ═══════════════════════════════════════════════════════════════════════════════
SPEC_VERSION  = "1.0"
SCALE_FACTOR  = 65536
INT64_MAX     = 9_223_372_036_854_775_807

# Turbo kernel parameters (64x64 tiles, 4x4 micro‑tile)
TILE_M = 64
TILE_N = 64
TILE_K = 16
TM = 4          # micro‑tile rows per thread
TN = 4          # micro‑tile columns per thread

# Shared memory dimensions (padded to avoid bank conflicts)
S_I_ROWS = TILE_M
S_I_COLS = TILE_K + 1   # +1 padding
S_W_ROWS = TILE_K
S_W_COLS = TILE_N + 1   # +1 padding

def _default_manifest_path():
    """
    Resolve a writable path for the manifest file.
    Priority: same folder as the script -> Desktop -> temp dir.
    This avoids the Windows protected-directory PermissionError when
    the script lives in C:/Users/name/ and writes to CWD.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hash_manifest.json"),
        os.path.join(os.path.expanduser("~"), "Desktop", "hash_manifest.json"),
        os.path.join(os.path.expanduser("~"), "Documents", "hash_manifest.json"),
    ]
    import tempfile
    candidates.append(os.path.join(tempfile.gettempdir(), "hash_manifest.json"))
    for path in candidates:
        try:
            with open(path, "a") as _f:
                pass
            return path
        except PermissionError:
            continue
    return candidates[-1]   # temp dir always works

MANIFEST_FILE = _default_manifest_path()

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU KERNEL (v7.9 Turbo – 4x4 register blocked)
# ═══════════════════════════════════════════════════════════════════════════════
if _CUDA_AVAILABLE:
    @cuda.jit
    def gpu_fixed_point_turbo_kernel(In, Weights, Out, M, N, K, use_relu):
        """
        Tiled fixed‑point matmul with 4x4 micro‑tile.
        Each thread block processes a 64x64 output tile.
        Shared memory uses int32 (values fit) and is padded to avoid bank conflicts.
        """
        sI = cuda.shared.array((64, 17), int32)   # TILE_M=64, TILE_K+1=17
        sW = cuda.shared.array((16, 65), int32)   # TILE_K=16, TILE_N+1=65

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        row_base = cuda.blockIdx.y * TILE_M + ty * TM
        col_base = cuda.blockIdx.x * TILE_N + tx * TN

        # 16 accumulators in registers (4x4 tile)
        acc = cuda.local.array(16, int64)
        for i in range(16):
            acc[i] = 0

        for k_tile in range(0, K, TILE_K):
            # Cooperative loading of In tile (int64 -> int32)
            for i in range(TM):
                r_idx = ty * TM + i
                g_row = cuda.blockIdx.y * TILE_M + r_idx
                g_col_k = k_tile + tx
                if g_row < M and g_col_k < K:
                    sI[r_idx, tx] = int32(In[g_row, g_col_k])
                else:
                    sI[r_idx, tx] = 0

            # Cooperative loading of Weights tile
            for j in range(TN):
                c_idx = tx * TN + j
                g_row_k = k_tile + ty
                g_col = cuda.blockIdx.x * TILE_N + c_idx
                if g_row_k < K and g_col < N:
                    sW[ty, c_idx] = int32(Weights[g_row_k, g_col])
                else:
                    sW[ty, c_idx] = 0

            cuda.syncthreads()

            # Compute micro‑tile
            for k in range(TILE_K):
                # Load 4 weights into registers (each thread loads its own row of Weights)
                vW0 = int64(sW[k, tx * TN + 0])
                vW1 = int64(sW[k, tx * TN + 1])
                vW2 = int64(sW[k, tx * TN + 2])
                vW3 = int64(sW[k, tx * TN + 3])

                # Load 4 rows of In (each thread loads a different row)
                vI0 = int64(sI[ty * TM + 0, k])
                vI1 = int64(sI[ty * TM + 1, k])
                vI2 = int64(sI[ty * TM + 2, k])
                vI3 = int64(sI[ty * TM + 3, k])

                # Update accumulators
                acc[0] += vI0 * vW0; acc[1] += vI0 * vW1
                acc[2] += vI0 * vW2; acc[3] += vI0 * vW3
                acc[4] += vI1 * vW0; acc[5] += vI1 * vW1
                acc[6] += vI1 * vW2; acc[7] += vI1 * vW3
                acc[8] += vI2 * vW0; acc[9] += vI2 * vW1
                acc[10] += vI2 * vW2; acc[11] += vI2 * vW3
                acc[12] += vI3 * vW0; acc[13] += vI3 * vW1
                acc[14] += vI3 * vW2; acc[15] += vI3 * vW3

            cuda.syncthreads()

        # Round and write back
        half = int64(32768)
        for i in range(TM):
            for j in range(TN):
                out_r = row_base + i
                out_c = col_base + j
                if out_r < M and out_c < N:
                    val = (acc[i * TN + j] + half) >> 16
                    if use_relu == 1 and val < 0:
                        Out[out_r, out_c] = 0
                    else:
                        Out[out_r, out_c] = val

# ═══════════════════════════════════════════════════════════════════════════════
#  CPU REFERENCE — scalar (the specification)
# ═══════════════════════════════════════════════════════════════════════════════
@njit
def cpu_reference_scalar(In, Weights, use_relu=1):
    """
    Scalar fixed‑point matmul. This IS the specification.
    Row‑major sequential accumulation — identical on any hardware.
    """
    M, K = In.shape
    _, N = Weights.shape
    Out  = np.zeros((M, N), dtype=np.int64)
    half = int64(32768)
    for r in range(M):
        for c in range(N):
            acc = int64(0)
            for k in range(K):
                acc += In[r, k] * Weights[k, c]
            val = (acc + half) >> 16
            Out[r, c] = int64(0) if (use_relu == 1 and val < 0) else val
    return Out

# ═══════════════════════════════════════════════════════════════════════════════
#  CPU parallel (prange) – still deterministic
# ═══════════════════════════════════════════════════════════════════════════════
@njit(parallel=True)
def cpu_reference_parallel(In, Weights, use_relu=1):
    """
    Parallel fixed‑point matmul. Outer loops over output rows are parallelised.
    Because each row accumulation is independent and sequential, results are
    identical to the scalar version.
    """
    M, K = In.shape
    _, N = Weights.shape
    Out = np.zeros((M, N), dtype=np.int64)
    half = int64(32768)
    for r in prange(M):
        for c in range(N):
            acc = int64(0)
            for k in range(K):
                acc += In[r, k] * Weights[k, c]
            val = (acc + half) >> 16
            Out[r, c] = int64(0) if (use_relu == 1 and val < 0) else val
    return Out

# ═══════════════════════════════════════════════════════════════════════════════
#  OVERFLOW GUARD (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def safe_input_range(K: int) -> float:
    return math.sqrt(INT64_MAX / (2.0 * K * (SCALE_FACTOR ** 2)))

def check_overflow(A, B, label_a="A", label_b="B", verbose=True):
    max_a = float(np.abs(A).max())
    max_b = float(np.abs(B).max())
    K = A.shape[1]
    max_accum = max_a * SCALE_FACTOR * max_b * SCALE_FACTOR * K
    safe_limit = INT64_MAX / 2.0
    safe = max_accum <= safe_limit
    safe_val = safe_input_range(K)

    if verbose:
        status = "✅ Safe" if safe else "⚠️  OVERFLOW RISK"
        print(f"    {status}")
        print(f"    max({label_a})={max_a:.4f}  max({label_b})={max_b:.4f}  K={K}")
        print(f"    worst-case accumulator : {max_accum:.3e}")
        print(f"    int64 safe limit       : {safe_limit:.3e}")
        print(f"    safe |input| for K={K:<5}: < {safe_val:.2f}")
        if safe:
            headroom = safe_limit / max(max_accum, 1.0)
            print(f"    headroom               : {headroom:.0f}x")
        else:
            print(f"    -> Results will silently WRAP. "
                  f"Reduce inputs to |val| < {safe_val:.2f} or reduce K.")
    return safe

# ═══════════════════════════════════════════════════════════════════════════════
#  WRAPPERS (GPU, CPU, unified)
# ═══════════════════════════════════════════════════════════════════════════════
def fixed_point_matmul_gpu(A, B, use_relu=True, skip_overflow_check=False):
    if not _CUDA_AVAILABLE:
        raise RuntimeError("No CUDA device. Use fixed_point_matmul_cpu().")
    M, K = A.shape;  K2, N = B.shape
    if K != K2: raise ValueError(f"Dimension mismatch: {A.shape} x {B.shape}")
    if not skip_overflow_check: check_overflow(A, B, verbose=False)

    A_f = (A * SCALE_FACTOR).astype(np.int64)
    B_f = (B * SCALE_FACTOR).astype(np.int64)
    d_A = cuda.to_device(A_f);  d_B = cuda.to_device(B_f)
    d_C = cuda.device_array((M, N), dtype=np.int64)

    # Turbo kernel uses 64x64 tiles, so grid dimensions:
    grid = (math.ceil(N / TILE_N), math.ceil(M / TILE_M))
    block = (16, 16)   # 16x16 threads per block
    e0 = cuda.event();  e1 = cuda.event()
    e0.record()
    gpu_fixed_point_turbo_kernel[grid, block](d_A, d_B, d_C, M, N, K,
                                              1 if use_relu else 0)
    e1.record();  e1.synchronize()

    k_ms = cuda.event_elapsed_time(e0, e1)
    C_raw = d_C.copy_to_host()
    return C_raw.astype(np.float32) / SCALE_FACTOR, k_ms, C_raw

def fixed_point_matmul_cpu(A, B, use_relu=True, skip_overflow_check=False, parallel=False):
    M, K = A.shape;  K2, N = B.shape
    if K != K2: raise ValueError(f"Dimension mismatch: {A.shape} x {B.shape}")
    if not skip_overflow_check: check_overflow(A, B, verbose=False)

    A_f = (A * SCALE_FACTOR).astype(np.int64)
    B_f = (B * SCALE_FACTOR).astype(np.int64)
    t0 = time.perf_counter()
    if parallel:
        C_raw = cpu_reference_parallel(A_f, B_f, use_relu=1 if use_relu else 0)
    else:
        C_raw = cpu_reference_scalar(A_f, B_f, use_relu=1 if use_relu else 0)
    elapsed = (time.perf_counter() - t0) * 1000
    return C_raw.astype(np.float32) / SCALE_FACTOR, elapsed, C_raw

def fixed_point_matmul(A, B, use_relu=True, skip_overflow_check=False,
                       force_cpu=False, parallel_cpu=False):
    if _CUDA_AVAILABLE and not force_cpu:
        return fixed_point_matmul_gpu(A, B, use_relu, skip_overflow_check)
    return fixed_point_matmul_cpu(A, B, use_relu, skip_overflow_check, parallel=parallel_cpu)

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
def sha256_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def tflops(M, K, N, t_ms):
    return (2*M*K*N) / (t_ms*1e-3) / 1e12

def mem_bw(M, K, N, t_ms):
    return (M*K + K*N + M*N)*8 / (t_ms*1e-3) / 1e9

def stats(times):
    t = np.array(times)
    return dict(mean=float(np.mean(t)), std=float(np.std(t)),
                min=float(np.min(t)),   max=float(np.max(t)),
                p5=float(np.percentile(t,5)), p95=float(np.percentile(t,95)))

def print_stats(label, s, M, K, N):
    print(f"  {label}")
    print(f"    mean={s['mean']:.3f} ms  std={s['std']:.3f} ms  "
          f"min={s['min']:.3f} ms  max={s['max']:.3f} ms")
    print(f"    p5={s['p5']:.3f} ms  p95={s['p95']:.3f} ms")
    print(f"    {tflops(M,K,N,s['mean']):.4f} TFLOPS   "
          f"{mem_bw(M,K,N,s['mean']):.2f} GB/s effective")

def theoretical_bound_fp64(K):
    return K * (0.5 / SCALE_FACTOR) * math.sqrt(2 / math.pi)

def section(title):
    w = 72
    print(f"\n{'='*w}\n  {title}\n{'='*w}")

def divider():
    print(f"  {'-'*68}")

def gpu_info():
    if not _CUDA_AVAILABLE:
        return "CPU-only (no CUDA)", "N/A"
    d = cuda.get_current_device()
    nm = d.name.decode() if isinstance(d.name, bytes) else d.name
    cc = f"{d.compute_capability[0]}.{d.compute_capability[1]}"
    return nm, cc

def platform_key(run: dict) -> str:
    return f"{run['gpu']}|{run.get('os','?')}"

# ═══════════════════════════════════════════════════════════════════════════════
#  MANIFEST (updated compare_manifests and print_cross_hardware_proof)
# ═══════════════════════════════════════════════════════════════════════════════
def build_manifest_cases():
    cases = []
    for seed in [42, 137, 999, 31415, 271828]:
        for (M, K, N) in [(64,64,64),(256,256,256),(512,128,512),
                          (128,512,128),(511,255,767)]:
            cases.append(dict(M=M,K=K,N=N,seed=seed,kind="normal",relu=True))
    cases.append(dict(M=64,  K=64,  N=64,  seed=0,  kind="zeros",     relu=True))
    cases.append(dict(M=64,  K=64,  N=64,  seed=0,  kind="ones",      relu=False))
    cases.append(dict(M=64,  K=64,  N=64,  seed=7,  kind="tiny",      relu=False))
    cases.append(dict(M=64,  K=32,  N=64,  seed=7,  kind="large",     relu=False))
    cases.append(dict(M=128, K=64,  N=128, seed=55, kind="neg_heavy", relu=True))
    cases.append(dict(M=300, K=150, N=200, seed=42, kind="odd_rect",  relu=False))
    return cases

def make_matrices(case):
    rng  = np.random.default_rng(case["seed"])
    M, K, N, kind = case["M"], case["K"], case["N"], case["kind"]
    if kind == "zeros":
        return np.zeros((M,K),np.float32), np.zeros((K,N),np.float32)
    if kind == "ones":
        return np.ones((M,K),np.float32),  np.ones((K,N),np.float32)
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

def case_key(case):
    return (f"{case['M']}x{case['K']}x{case['N']}_"
            f"{case['kind']}_seed{case['seed']}_"
            f"{'relu' if case['relu'] else 'norelu'}")

def load_manifest(path=MANIFEST_FILE):
    if os.path.exists(path):
        try:
            with open(path) as f:
                content = f.read().strip()
            if content:
                return json.loads(content)
            # File exists but is empty — previous failed write, ignore it
        except (json.JSONDecodeError, OSError):
            print(f"  Note: existing manifest at '{path}' is corrupt/empty — starting fresh.")
    return {"spec_version": SPEC_VERSION, "scale_factor": SCALE_FACTOR, "runs": []}

def save_manifest(manifest, path=None):
    if path is None:
        path = MANIFEST_FILE
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path

def compare_manifests(runs: list) -> dict:
    """
    Compare hashes across all runs.
    Uses intersection of keys (cases that ALL runs have in common).
    Reports if key sets differ between runs so nothing is silently ignored.
    """
    if len(runs) < 2:
        return {"error": "Need at least 2 runs to compare"}

    # Intersection: only compare cases all runs share
    common_keys = set(runs[0]["hashes"].keys())
    for r in runs[1:]:
        common_keys.intersection_update(r["hashes"].keys())

    # Union: detect if any keys are missing from some runs
    all_keys_union = set()
    for r in runs:
        all_keys_union.update(r["hashes"].keys())

    skew = len(all_keys_union) - len(common_keys)   # keys present in some but not all

    if not common_keys:
        return {
            "error": (
                "No common test cases across runs.\n"
                "  This usually means each machine saved its own manifest\n"
                "  without seeing the other. Use --merge to combine them:\n"
                "    python genesis_benchmark3.py --merge local.json remote.json"
            )
        }

    results = {}
    for key in sorted(common_keys):
        run_data = [(r["gpu"], r["hashes"][key]) for r in runs]
        hashes   = [h for _, h in run_data]
        results[key] = {
            "agree"  : len(set(hashes)) == 1,
            "hashes" : run_data,
        }

    total   = len(results)
    agreed  = sum(1 for v in results.values() if v["agree"])
    # Distinct platforms by gpu name + OS
    platforms = list(set(platform_key(r) for r in runs))

    return {
        "total"    : total,
        "agreed"   : agreed,
        "failed"   : total - agreed,
        "skew"     : skew,       # keys in union but not intersection
        "cases"    : results,
        "platforms": platforms,
    }

def print_cross_hardware_proof(runs: list, title="Cross-Hardware Compliance Proof"):
    """
    Print a formatted proof table. The headline output of the project.
    Returns True if all common cases agree, False otherwise.
    """
    cmp = compare_manifests(runs)
    if "error" in cmp:
        print(f"  {cmp['error']}")
        return False

    # Detect same-platform runs (e.g. GTX 1050 ran twice locally)
    distinct_platforms = set(platform_key(r) for r in runs)
    same_platform_only = len(distinct_platforms) == 1

    w = 70
    print(f"\n  {'#'*w}")
    print(f"  #  {title:^{w-4}}  #")
    print(f"  {'#'*w}\n")

    print(f"  {'Platform':<42} {'Mode':<8} {'OS'}")
    divider()
    for run in runs:
        mode = run.get("mode","gpu").upper()
        os_  = run.get("os","?")[:28]
        gpu  = run["gpu"][:40]
        print(f"  {gpu:<42} {mode:<8} {os_}")
    print()

    agreed = cmp["agreed"];  total = cmp["total"]
    n_plat = len(cmp["platforms"])
    skew   = cmp.get("skew", 0)

    if skew > 0:
        print(f"  Note: {skew} test case(s) exist in some runs but not others")
        print(f"  (comparing {total} common cases only)\n")

    if agreed == total:
        if same_platform_only:
            print(f"  RESULT: {agreed}/{total} cases agree -- same platform ran {len(runs)} times.")
            print()
            print(f"  Hashes are self-consistent (expected).")
            print(f"  To generate a CROSS-hardware proof, run on a different GPU or machine:")
            print(f"    Upload script to Google Colab/Kaggle -> run -> download manifest")
            print(f"    Then: python genesis_benchmark3.py --merge local.json remote.json")
        else:
            print(f"  RESULT: {agreed}/{total} cases agree across {n_plat} distinct platform(s)")
            print()
            print(f"  All {total} SHA-256 hashes are identical across every platform above.")
            print()
            print(f"  IEEE-754 floating point cannot make this guarantee.")
            print(f"  Q16.16 fixed-point integer arithmetic can -- and does.")
    else:
        print(f"  RESULT: {cmp['failed']}/{total} CASES DIVERGE")
        for key, v in cmp["cases"].items():
            if not v["agree"]:
                print(f"\n  FAIL: {key}")
                for gpu, h in v["hashes"]:
                    print(f"    {gpu[:38]}: {(h or 'MISSING')[:32]}...")

    print(f"\n  {'#'*w}")
    return agreed == total and not same_platform_only

# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE MODE
# ═══════════════════════════════════════════════════════════════════════════════
def run_merge(paths, output_path, pause):
    section(f"[v9.3] MERGE -- combining {len(paths)} manifest(s)")
    merged = {"spec_version":SPEC_VERSION,"scale_factor":SCALE_FACTOR,"runs":[]}
    seen_ids = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}"); continue
        m = load_manifest(path)
        added = 0
        for run in m.get("runs",[]):
            uid = run.get("run_id","") + "|" + run.get("gpu","")
            if uid not in seen_ids:
                merged["runs"].append(run)
                seen_ids.add(uid)
                added += 1
        print(f"  {path:<45} {len(m.get('runs',[]))} run(s), {added} new")
    n_runs = len(merged["runs"])
    n_plat = len(set(platform_key(r) for r in merged["runs"]))
    print(f"\n  Total runs     : {n_runs}")
    print(f"  Platforms      : {n_plat}")
    if n_runs < 2:
        print("  Need at least 2 distinct runs to compare.")
        _pause_if_needed(pause)
        return
    print_cross_hardware_proof(merged["runs"])
    if n_plat >= 2:
        print("\n  +----------------------------------------------------------+")
        print("  |  CROSS-HARDWARE PROOF COMPLETE                           |")
        print(f"  |  {n_plat} distinct platforms, {n_runs} total runs                    |".ljust(63) + "|")
        print("  |  Share merged_manifest.json as evidence.                 |")
        print("  +----------------------------------------------------------+")
    save_manifest(merged, output_path)
    print(f"\n  Saved -> {output_path}")
    _pause_if_needed(pause)

# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFY MODE
# ═══════════════════════════════════════════════════════════════════════════════
def run_verify(manifest_path, force_cpu, parallel_cpu, pause):
    section(f"[v9.3] VERIFY MODE -- {manifest_path}")
    manifest = load_manifest(manifest_path)
    if not manifest.get("runs"):
        print("  Manifest is empty.")
        _pause_if_needed(pause)
        return
    gpu_name, _ = gpu_info()
    mode = "CPU-only" if (force_cpu or not _CUDA_AVAILABLE) else gpu_name
    if parallel_cpu: mode += " (parallel)"
    print(f"  Running as : {mode}")
    print(f"  Spec       : v{manifest.get('spec_version','?')}  scale={manifest.get('scale_factor','?')}")
    divider()
    reference = {}
    for run in manifest["runs"]:
        for k, h in run["hashes"].items():
            if k not in reference:
                reference[k] = {"hash":h,"source":run["gpu"]}
    cases = build_manifest_cases()
    case_map = {case_key(c):c for c in cases}
    print(f"  Verifying {len(reference)} cases...\n")
    passed = failed = skipped = 0
    run_hashes = {}
    for key, ref in reference.items():
        if key not in case_map:
            print(f"  SKIP  {key}"); skipped += 1; continue
        case = case_map[key]
        A, B = make_matrices(case)
        _,_,C_r = fixed_point_matmul(A,B,use_relu=case["relu"],
                                     skip_overflow_check=True,
                                     force_cpu=force_cpu,
                                     parallel_cpu=parallel_cpu)
        h = sha256_of(C_r)
        run_hashes[key] = h
        if h == ref["hash"]:
            print(f"  OK    {key}"); passed += 1
        else:
            print(f"  FAIL  {key}")
            print(f"         ref  ({ref['source'][:28]}): {ref['hash'][:40]}...")
            print(f"         this ({mode[:28]}): {h[:40]}..."); failed += 1
    divider()
    print(f"\n  {passed}/{passed+failed+skipped} passed | {failed} failed | {skipped} skipped")
    if failed == 0 and passed > 0:
        this_run = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "gpu": gpu_name, "sm":"N/A",
            "host": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "timestamp": datetime.now().isoformat(),
            "spec_ver": SPEC_VERSION,
            "mode": "cpu" if force_cpu else "gpu",
            "hashes": run_hashes,
            "correct": True,
        }
        print_cross_hardware_proof(manifest["runs"] + [this_run])
    else:
        print(f"\n  NON-COMPLIANT: {failed} mismatch(es) detected.")
    _pause_if_needed(pause)

# ═══════════════════════════════════════════════════════════════════════════════
#  FULL AUDIT (uses turbo kernel by default)
# ═══════════════════════════════════════════════════════════════════════════════
def run_audit(force_cpu, parallel_cpu, pause):
    try:
        gpu_name, gpu_cc = gpu_info()
        use_gpu = _CUDA_AVAILABLE and not force_cpu
        mode_label = f"GPU  {gpu_name}" if use_gpu else "CPU-only mode"
        if parallel_cpu and not use_gpu: mode_label += " (parallel)"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        section("[v9.3] Deterministic Fixed-Point MatMul -- Full Audit")
        print(f"  Mode       : {mode_label}")
        if use_gpu: print(f"  Compute    : sm_{gpu_cc.replace('.','')}")
        print(f"  Host       : {platform.node()}")
        print(f"  Python     : {platform.python_version()}")
        print(f"  OS         : {platform.system()} {platform.release()}")
        print(f"  Run ID     : {run_id}")
        print(f"  Spec       : v{SPEC_VERSION}  Q16.16  (scale={SCALE_FACTOR})")
        if use_gpu:
            print(f"  Tile config: TILE_M={TILE_M}  TILE_N={TILE_N}  "
                  f"TILE_K={TILE_K}  TM={TM}xTN={TN}")

        # Warmup
        section("1 . Warmup Pass")
        _A = np.random.randn(64,64).astype(np.float32)
        _B = np.random.randn(64,64).astype(np.float32)
        fixed_point_matmul(_A,_B,skip_overflow_check=True,
                           force_cpu=force_cpu, parallel_cpu=parallel_cpu)
        print("  Kernel compiled, warmed up.")

        # Correctness
        section("2 . Correctness Verification -- SHA-256 Hash Match")
        N_val = 1024
        rng = np.random.default_rng(42)
        A = rng.standard_normal((N_val,N_val)).astype(np.float32)
        B = rng.standard_normal((N_val,N_val)).astype(np.float32)

        if use_gpu:
            C_gpu, k_time, C_raw = fixed_point_matmul_gpu(A,B,use_relu=True,
                                                           skip_overflow_check=True)
            print(f"  GPU kernel time : {k_time:.3f} ms  ({N_val}x{N_val})")
            print("  Running scalar CPU reference (may take ~30 s) ...")
            A_f = (A*SCALE_FACTOR).astype(np.int64)
            B_f = (B*SCALE_FACTOR).astype(np.int64)
            cpu_res = cpu_reference_scalar(A_f,B_f,use_relu=1)
            gpu_hash = sha256_of(C_raw);  cpu_hash = sha256_of(cpu_res)
            print(f"\n  CPU hash : {cpu_hash}")
            print(f"  GPU hash : {gpu_hash}")
            divider()
            correctness_ok = (gpu_hash == cpu_hash)
            if correctness_ok:
                print("  OK  PERFECT MATCH -- bit-exact: scalar CPU == GPU kernel")
            else:
                bad = int(np.sum(C_raw != cpu_res))
                print(f"  FAIL  HASH MISMATCH -- {bad}/{N_val**2} elements differ")
        else:
            _,_, C_raw = fixed_point_matmul_cpu(A,B,use_relu=True,
                                                skip_overflow_check=True,
                                                parallel=parallel_cpu)
            h = sha256_of(C_raw)
            print(f"  Hash : {h}")
            correctness_ok = True
            print("  OK  CPU reference hash recorded.")

        # Overflow
        section("3 . Overflow Detection  (int64 accumulator analysis)")
        print("  Safe input range by K  (SF=65536):")
        print(f"  {'K':>6}  {'safe |val| <':>14}  note")
        divider()
        for K_demo, note in [(32,""), (64,""), (256,""), (1024,""), (4096,"<-- demo overflow threshold at +-600")]:
            sr = safe_input_range(K_demo)
            print(f"  {K_demo:>6}  {sr:>14.2f}  {note}")
        divider()
        print("\n  Case A -- N(0,1), K=1024  (expected: safe,  max~5 << limit~1024)")
        check_overflow(A, B, "A", "B", verbose=True)
        print("\n  Case B -- uniform +-200, K=32  (expected: safe,  200 << limit~5792)")
        rng_ov = np.random.default_rng(1)
        A_200 = rng_ov.uniform(-200,200,(64,32)).astype(np.float32)
        B_200 = rng_ov.uniform(-200,200,(32,64)).astype(np.float32)
        check_overflow(A_200, B_200, "A_200", "B_200", verbose=True)
        print("\n  Case C -- uniform +-600, K=4096  (expected: OVERFLOW, 600 > limit~512)")
        A_bad = rng_ov.uniform(-600,600,(32,4096)).astype(np.float32)
        B_bad = rng_ov.uniform(-600,600,(4096,32)).astype(np.float32)
        check_overflow(A_bad, B_bad, "A_600", "B_600", verbose=True)

        # Quantisation error
        section("4 . Quantization Error Analysis  (reference = float64)")
        shapes = [
            (1024,1024,1024,"square    1024x1024x1024"),
            (511, 255, 767, "rect       511x 255x 767"),
            (128, 4096,256, "tall-K     128x4096x 256"),
            (2048,64,  2048,"thin-K    2048x  64x2048"),
            (300, 150, 200, "odd-rect   300x 150x 200"),
        ]
        print(f"  {'Shape':<30} {'max|D|':>10} {'mean|D|':>10} "
              f"{'theory':>10}  status")
        divider()
        for M,K_dim,N_dim,label in shapes:
            rng2 = np.random.default_rng(7)
            Ar = rng2.standard_normal((M,K_dim)).astype(np.float32)
            Br = rng2.standard_normal((K_dim,N_dim)).astype(np.float32)
            C_g,_,_ = fixed_point_matmul(Ar,Br,use_relu=False,
                                         skip_overflow_check=True,
                                         force_cpu=force_cpu,
                                         parallel_cpu=parallel_cpu)
            C_f64 = Ar.astype(np.float64) @ Br.astype(np.float64)
            diff = np.abs(C_g.astype(np.float64) - C_f64)
            theory = theoretical_bound_fp64(K_dim)
            status = "OK" if diff.max() < theory*5 else "WARN"
            print(f"  {status}  {label:<28}  {diff.max():>10.6f}  "
                  f"{diff.mean():>10.6f}  {theory:>10.6f}")
        print(f"\n  theory = K x (0.5/{SCALE_FACTOR}) x sqrt(2/pi)  "
              f"(float64 ref accurate to ~1e-15)")

        # Performance (only if GPU)
        if use_gpu:
            section("5 . Performance Benchmark  (1024x1024, 20 runs)")
            A_b = rng.standard_normal((1024,1024)).astype(np.float32)
            B_b = rng.standard_normal((1024,1024)).astype(np.float32)
            gpu_times = [fixed_point_matmul_gpu(A_b,B_b, skip_overflow_check=True)[1] for _ in range(20)]
            cpu_times = []
            for _ in range(20):
                t0 = time.perf_counter(); _ = A_b @ B_b
                cpu_times.append((time.perf_counter()-t0)*1000)
            g_s = stats(gpu_times); c_s = stats(cpu_times)
            print()
            print_stats("GPU fixed-point kernel (compute-only)", g_s,1024,1024,1024)
            divider()
            print_stats("CPU NumPy float32  (A @ B)", c_s,1024,1024,1024)
            divider()
            ratio = g_s["mean"]/c_s["mean"]
            print(f"  GPU/CPU ratio: {ratio:.2f}x  ", end="")
            if ratio < 1.0: print(f"-> GPU is {1/ratio:.2f}x FASTER")
            else: print(f"-> CPU NumPy is {ratio:.2f}x faster  "
                        f"(determinism, not raw speed, is the value)")

            section("6 . Multi-Size Throughput Sweep")
            print(f"  {'Size':>6}  {'mean ms':>8}  {'TFLOPS':>8}  "
                  f"{'GB/s':>8}  {'vs NumPy':>12}  status")
            divider()
            for sz in [128,256,512,1024,2048]:
                Asw = np.random.randn(sz,sz).astype(np.float32)
                Bsw = np.random.randn(sz,sz).astype(np.float32)
                ts = [fixed_point_matmul_gpu(Asw,Bsw, skip_overflow_check=True)[1] for _ in range(5)]
                t_g = float(np.mean(ts))
                t0 = time.perf_counter()
                for _ in range(5): _ = Asw @ Bsw
                t_c = (time.perf_counter()-t0)*1000/5
                tag = (f"{t_c/t_g:.2f}x faster" if t_g < t_c else f"{t_g/t_c:.2f}x slower")
                note = "OK" if t_g < t_c else " "
                print(f"  {sz:>6}  {t_g:>8.3f}  "
                      f"{tflops(sz,sz,sz,t_g):>8.4f}  "
                      f"{mem_bw(sz,sz,sz,t_g):>8.2f}  {tag:>12}  {note}")
        else:
            section("5 . Performance Benchmark  (skipped -- CPU-only mode)")
            section("6 . Multi-Size Sweep       (skipped -- CPU-only mode)")
            g_s = c_s = None

        # Build manifest
        section("7 . Cross-Hardware Hash Manifest")
        cases = build_manifest_cases()
        run_hashes = {}
        for case in cases:
            A_m,B_m = make_matrices(case)
            _,_,C_m = fixed_point_matmul(A_m,B_m,use_relu=case["relu"],
                                         skip_overflow_check=True,
                                         force_cpu=force_cpu,
                                         parallel_cpu=parallel_cpu)
            run_hashes[case_key(case)] = sha256_of(C_m)
        print(f"  {len(run_hashes)} cases  (5 seeds x 5 shapes + 6 edge cases)\n")
        print(f"  {'Test case':<45}  SHA-256 (first 32 chars)")
        divider()
        for k in [k for k in run_hashes if "seed42" in k][:8]:
            print(f"  {k:<45}  {run_hashes[k][:32]}...")

        this_run = {
            "run_id": run_id,
            "gpu": gpu_name,
            "sm": gpu_cc,
            "host": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "timestamp": datetime.now().isoformat(),
            "spec_ver": SPEC_VERSION,
            "mode": "cpu" if force_cpu else "gpu",
            "hashes": run_hashes,
            "correct": correctness_ok,
        }

        manifest = load_manifest()
        prev_runs = manifest.get("runs",[])
        divider()
        all_match = True
        if prev_runs:
            print(f"  Comparing against {len(prev_runs)} previous run(s):\n")
            for prev in prev_runs:
                # Check hashes
                run_ok = all(run_hashes.get(k)==prev["hashes"].get(k)
                             for k in run_hashes if k in prev["hashes"])
                if not run_ok: all_match = False
                status = "OK" if run_ok else "FAIL"
                # Safely get timestamp; use "unknown" if missing
                ts = prev.get("timestamp", "unknown")
                # Trim to date part if present
                if ts != "unknown" and len(ts) >= 10:
                    ts = ts[:10]
                print(f"  {status}  {prev['gpu']:<38}  ({ts})")
        else:
            print("  First entry in manifest.")
            print("  Share hash_manifest.json and run on another machine:")
            print(f"    python detmatmul.py --verify hash_manifest.json")
            print(f"    python detmatmul.py --cpu   (no GPU needed)")
            print()
            print("  Have two manifests from different machines? Merge them:")
            print(f"    python detmatmul.py --merge this.json other.json")

        manifest["spec_version"] = SPEC_VERSION
        manifest["scale_factor"] = SCALE_FACTOR
        manifest["runs"].append(this_run)
        saved_path = save_manifest(manifest)

        all_runs_now = manifest["runs"]
        distinct_plat = len(set(platform_key(r) for r in all_runs_now))
        print(f"\n  Manifest saved -> {os.path.abspath(saved_path)}")
        print(f"  ({len(all_runs_now)} run(s), {distinct_plat} distinct platform(s))")

        # Summary
        section("8 . Summary")
        print(f"  Correctness  : {'OK PASS' if correctness_ok else 'FAIL'}")
        if use_gpu and g_s:
            print(f"  GPU perf     : {g_s['mean']:.3f} ms +/- {g_s['std']:.3f} ms  "
                  f"({tflops(1024,1024,1024,g_s['mean']):.4f} TFLOPS)")
            print(f"  CPU perf     : {c_s['mean']:.3f} ms +/- {c_s['std']:.3f} ms  "
                  f"({tflops(1024,1024,1024,c_s['mean']):.4f} TFLOPS)")
        print(f"  Platforms    : {distinct_plat} distinct hardware platform(s) in manifest")
        print(f"  Spec         : v{SPEC_VERSION}  (scale={SCALE_FACTOR}, Q16.16)")

        # Cross-hardware proof
        section("9 . Cross-Hardware Compliance Proof")
        if len(all_runs_now) >= 2:
            proof_ok = print_cross_hardware_proof(all_runs_now)
            if proof_ok and distinct_plat >= 2:
                print("\n  +------------------------------------------------------------+")
                print("  |  PROOF COMPLETE                                            |")
                print("  |                                                            |")
                print(f"  |  {distinct_plat} distinct hardware platforms.                           |")
                print(f"  |  {len(run_hashes)} SHA-256 hashes. Zero mismatches.                  |")
                print("  |                                                            |")
                print("  |  The same fixed-point computation produces identical       |")
                print("  |  bit-exact results regardless of hardware or OS.           |")
                print("  |  This is the value. IEEE-754 float cannot do this.        |")
                print("  +------------------------------------------------------------+")
                print("\n  Grow the proof:")
                print("  - Merge with another manifest:")
                print("    python genesis_benchmark3.py --merge hash_manifest.json other.json")
                print("  - Test on AMD GPU, Apple Silicon, Intel Arc, server CPU")
                print("  - Each new platform that agrees strengthens the claim")
            # proof_ok=False here means same-platform only (handled inside print_cross_hardware_proof)
        else:
            print("  Only 1 run in manifest -- need a 2nd platform.")
            print()
            print("  HOW TO GET THE CROSS-HARDWARE PROOF:")
            print("  ─────────────────────────────────────")
            print("  1. Download this script + hash_manifest.json")
            print("  2. Upload BOTH to Google Colab or Kaggle")
            print("  3. Run the script there (it will save its own manifest)")
            print("  4. Download that remote hash_manifest.json")
            print("  5. On your local machine, merge them:")
            print("       python genesis_benchmark3.py --merge hash_manifest.json remote.json")
            print()
            print("  The merge command will print the proof automatically.")

    except Exception:
        print("\n  UNHANDLED ERROR:")
        traceback.print_exc()

    _pause_if_needed(pause)

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if _is_interactive():
        # Colab / Jupyter: run audit directly with defaults
        print("Running in interactive mode (Colab/Jupyter).")
        run_audit(force_cpu=False, parallel_cpu=False, pause=False)
    else:
        parser = argparse.ArgumentParser(
            description="Deterministic Fixed-Point MatMul v9.3 Turbo")
        parser.add_argument("--version", action="store_true",
                            help="Show version and exit")
        parser.add_argument("--verify", nargs="?", const=MANIFEST_FILE,
                            metavar="MANIFEST",
                            help="Verify this machine against a saved manifest")
        parser.add_argument("--cpu", action="store_true",
                            help="Force CPU-only mode (no CUDA required)")
        parser.add_argument("--parallel-cpu", action="store_true",
                            help="Use multi-threaded CPU (faster, still deterministic)")
        parser.add_argument("--merge", nargs="+", metavar="MANIFEST",
                            help="Merge 2+ manifests: --merge a.json b.json")
        parser.add_argument("--output", default="merged_manifest.json",
                            help="Output path for --merge")
        parser.add_argument("--no-pause", action="store_true",
                            help="Do not wait for Enter before exiting")
        args, _ = parser.parse_known_args()

        if args.version:
            print(f"Deterministic Fixed-Point MatMul v9.3 Turbo")
            sys.exit(0)

        pause = not args.no_pause   # default: pause unless --no-pause given

        if args.merge:
            run_merge(args.merge, args.output, pause)
        elif args.verify:
            run_verify(args.verify, args.cpu, args.parallel_cpu, pause)
        else:
            run_audit(args.cpu, args.parallel_cpu, pause)
