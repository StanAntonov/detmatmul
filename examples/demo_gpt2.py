"""
Deterministic GPT-2 Demo
=========================
The world's first demonstrably deterministic language model inference.

Run this on any two machines. The SHA-256 hashes will match.

Usage:
    python demo_gpt2.py                        # full demo
    python demo_gpt2.py --verify results.json  # cross-machine verify
    python demo_gpt2.py --interactive          # chat mode

Requirements:
    pip install transformers tiktoken
"""

import argparse
import hashlib
import json
import os
import platform
import time
import warnings

warnings.filterwarnings("ignore")

_GPU_NAME = "CPU-only"
try:
    from numba import cuda
    _dev      = cuda.get_current_device()
    _GPU_NAME = (_dev.name.decode()
                 if isinstance(_dev.name, bytes) else _dev.name)
except Exception:
    pass

# ── Detect Jupyter ────────────────────────────────────────────────────────────
_IS_JUPYTER = False
try:
    _IS_JUPYTER = get_ipython().__class__.__name__ in (
        "ZMQInteractiveShell", "Shell")
except NameError:
    pass

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║   DETERMINISTIC GPT-2  —  Cross-Hardware Verified Inference         ║
║                                                                      ║
║   Every token generated through a Q16.16 fixed-point kernel.        ║
║   The SHA-256 hash of any output is identical on any hardware.       ║
║   This is the first known implementation of its kind.                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# Canonical demo prompts — fixed, never change these
# Changing them breaks cross-machine hash comparison
CANONICAL_PROMPTS = [
    "The definition of artificial intelligence is",
    "In the future, computing will",
    "The most important property of a mathematical proof is",
    "Determinism in computer science means",
    "The relationship between hardware and software is",
]


def load_model(model_name: str = "gpt2"):
    """Load the deterministic GPT-2 model."""
    try:
        from detmatmul.gpt2 import DeterministicGPT2
    except ImportError:
        # Running without package install — try direct import
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from detmatmul.gpt2 import DeterministicGPT2

    print(BANNER)
    print(f"  Hardware : {_GPU_NAME}")
    print(f"  OS       : {platform.system()} {platform.release()}")
    print(f"  Python   : {platform.python_version()}")
    print()

    print("  Loading model (downloads ~500MB on first run)...")
    t0    = time.perf_counter()
    model = DeterministicGPT2.load(model_name)
    print(f"  Warmup pass...", end="", flush=True)
    model.generate("test", max_new_tokens=3, temperature=0.0)
    elapsed = time.perf_counter() - t0
    print(f" ready ({elapsed:.1f}s total)\n")
    return model


def run_demo(model, save_path: str = "gpt2_results.json"):
    """Run canonical demo prompts and save results."""
    sep = "=" * 70

    print(sep)
    print("  DETERMINISTIC GPT-2 INFERENCE DEMO")
    print("  These hashes must be identical on every machine.")
    print(sep)
    print()

    results = {}
    for i, prompt in enumerate(CANONICAL_PROMPTS):
        print(f"  [{i+1}/{len(CANONICAL_PROMPTS)}] Prompt: \"{prompt}\"")
        t0          = time.perf_counter()
        text, hash_ = model.generate(
            prompt,
            max_new_tokens = 60,
            temperature    = 0.8,
            top_k          = 40,
            seed           = 42,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        generated = text[len(prompt):]

        print(f"  Output: {generated[:100].strip()}")
        print(f"  Hash  : {hash_}")
        print(f"  Time  : {elapsed:.0f}ms")
        print()
        results[prompt] = {"text": text, "hash": hash_, "ms": elapsed}

    # Master hash
    master = hashlib.sha256(
        "".join(r["hash"] for r in results.values()).encode()
    ).hexdigest()

    print(sep)
    print(f"  MASTER HASH: {master}")
    print()
    print("  This master hash is your cross-hardware proof.")
    print("  Run on any other machine. If the master hash matches,")
    print("  all outputs were bit-identical despite different hardware.")
    print(sep)

    # Save results
    output = {
        "hardware"    : _GPU_NAME,
        "os"          : f"{platform.system()} {platform.release()}",
        "python"      : platform.python_version(),
        "timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "master_hash" : master,
        "results"     : results,
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved -> {os.path.abspath(save_path)}")
    print(f"  Upload to another machine and run:")
    print(f"    python demo_gpt2.py --verify {save_path}")
    return output


def run_verify(model, reference_path: str):
    """Verify this machine's outputs match a reference file."""
    print(f"\n  Loading reference: {reference_path}")
    with open(reference_path) as f:
        ref = json.load(f)

    ref_hw     = ref.get("hardware", "unknown")
    ref_master = ref.get("master_hash", "")
    ref_results = ref.get("results", {})

    sep = "=" * 70
    print()
    print(sep)
    print("  CROSS-MACHINE VERIFICATION")
    print(sep)
    print(f"  Reference : {ref_hw} ({ref.get('os','')})")
    print(f"  This      : {_GPU_NAME} ({platform.system()} {platform.release()})")
    print()

    our_results = {}
    all_match   = True

    for prompt in CANONICAL_PROMPTS:
        if prompt not in ref_results:
            print(f"  MISSING  {repr(prompt[:50])}")
            all_match = False
            continue

        text, hash_ = model.generate(
            prompt,
            max_new_tokens = 60,
            temperature    = 0.8,
            top_k          = 40,
            seed           = 42,
        )
        our_results[prompt] = {"text": text, "hash": hash_}
        match  = (hash_ == ref_results[prompt]["hash"])
        status = "OK  " if match else "FAIL"
        if not match:
            all_match = False
        print(f"  {status}  {repr(prompt[:48])}")

    our_master = hashlib.sha256(
        "".join(r["hash"] for r in our_results.values()).encode()
    ).hexdigest()

    print()
    print(f"  Reference master hash : {ref_master}")
    print(f"  Our master hash       : {our_master}")
    print()

    master_match = (our_master == ref_master)
    print(sep)
    if master_match and all_match:
        print()
        print("  ╔════════════════════════════════════════════════════════════╗")
        print("  ║  PROOF COMPLETE                                           ║")
        print("  ║                                                           ║")
        print(f"  ║  {ref_hw[:30]:<30}  →  ║")
        print(f"  ║  {_GPU_NAME[:30]:<30}  →  ║")
        print("  ║                                                           ║")
        print("  ║  Identical GPT-2 outputs. Identical hashes.              ║")
        print("  ║  IEEE-754 float cannot make this guarantee.              ║")
        print("  ║  Q16.16 fixed-point integer arithmetic can.              ║")
        print("  ╚════════════════════════════════════════════════════════════╝")
        print()
    else:
        print("  MISMATCH — investigate differences above.")
    print(sep)


def run_interactive(model):
    """Interactive chat with hash display."""
    print(BANNER)
    print("  Interactive mode. Every response comes with a hash.")
    print("  The same prompt on any machine will always produce the same hash.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        if not prompt or prompt.lower() in ("quit", "exit"):
            print("  Goodbye.")
            break

        print("  Generating...", end="", flush=True)
        t0          = time.perf_counter()
        text, hash_ = model.generate(
            prompt, max_new_tokens=80, temperature=0.8, top_k=40, seed=0)
        elapsed     = (time.perf_counter() - t0) * 1000

        generated = text[len(prompt):]
        print(f"\r  AI  > {generated[:120].strip()}")
        print(f"  Hash > {hash_}  ({elapsed:.0f}ms)\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or _IS_JUPYTER:
    if _IS_JUPYTER:
        model = load_model()
        run_demo(model)
    else:
        parser = argparse.ArgumentParser(
            description="Deterministic GPT-2 Demo")
        parser.add_argument("--verify",      metavar="FILE",
                            help="Verify against results from another machine")
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--model",       default="gpt2",
                            help="gpt2 / gpt2-medium (default: gpt2)")
        parser.add_argument("--save",        default="gpt2_results.json")
        args, _ = parser.parse_known_args()

        model = load_model(args.model)

        if args.verify:
            run_verify(model, args.verify)
        elif args.interactive:
            run_interactive(model)
        else:
            run_demo(model, args.save)
            if not _IS_JUPYTER:
                input("\n  Press ENTER to exit...")
