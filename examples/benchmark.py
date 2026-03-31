"""
detmatmul benchmark
===================
Full audit, verify, and merge — as referenced in the README.

Usage:
    python examples/benchmark.py                              # full audit
    python examples/benchmark.py --cpu                       # CPU-only
    python examples/benchmark.py --parallel-cpu              # multi-threaded CPU
    python examples/benchmark.py --verify manifests/hash_manifest.json
    python examples/benchmark.py --merge a.json b.json
    python examples/benchmark.py --merge a.json b.json --output merged.json
"""

import sys
import os

# Allow running directly without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detmatmul.cli import main_audit, main_verify, main_merge

import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="detmatmul benchmark — audit, verify, and merge manifests.",
        add_help=False,
    )
    parser.add_argument("--verify",  metavar="MANIFEST", nargs="?", const="manifests/hash_manifest.json",
                        help="Verify this machine against a manifest")
    parser.add_argument("--merge",   metavar="MANIFEST", nargs="+",
                        help="Merge two or more manifests")
    parser.add_argument("--output",  default="merged_manifest.json",
                        help="Output path for --merge")
    parser.add_argument("--cpu",     action="store_true",
                        help="Force CPU-only mode (no CUDA required)")
    parser.add_argument("--parallel-cpu", action="store_true",
                        help="Use multi-threaded CPU kernel")
    parser.add_argument("--no-pause", action="store_true",
                        help="Do not wait for ENTER before exiting")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")

    args, _ = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    # Build forwarded argv for the sub-commands
    sub = []
    if args.cpu:          sub.append("--cpu")
    if args.parallel_cpu: sub.append("--parallel-cpu")
    if args.version:      sub.append("--version")

    if args.merge:
        main_merge(args.merge + ["--output", args.output])
    elif args.verify:
        main_verify([args.verify] + sub)
    else:
        main_audit(sub + (["--output", "hash_manifest.json"]))

    if not args.no_pause and sys.stdin.isatty():
        try:
            input("\nPress ENTER to exit...")
        except (EOFError, KeyboardInterrupt):
            pass

if __name__ == "__main__":
    main()
