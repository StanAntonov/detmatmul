"""
Tests for detmatmul.manifest
"""
import os
import json
import tempfile
import numpy as np
from detmatmul.manifest import (
    build_manifest, load_manifest, save_manifest,
    merge_manifests, compare_manifests,
)


def _make_run(gpu, hashes):
    return {"run_id": f"test_{gpu}", "gpu": gpu, "os": "TestOS",
            "hashes": hashes, "timestamp": "2026-01-01T00:00:00"}


def test_build_manifest_structure():
    m = build_manifest(force_cpu=True, gpu_name="TestCPU")
    assert "runs" in m
    assert len(m["runs"]) == 1
    assert len(m["runs"][0]["hashes"]) == 31


def test_save_load_roundtrip():
    m = build_manifest(force_cpu=True, gpu_name="TestCPU")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_manifest(m, path)
        m2 = load_manifest(path)
        assert m2["runs"][0]["hashes"] == m["runs"][0]["hashes"]
    finally:
        os.unlink(path)


def test_load_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        m = load_manifest(path)   # empty file — should return blank manifest
        assert m["runs"] == []
    finally:
        os.unlink(path)


def test_load_missing_file():
    m = load_manifest("/nonexistent/path/manifest.json")
    assert m["runs"] == []


def test_merge_deduplicates():
    h = {"key1": "abc"}
    r = _make_run("GPU_A", h)
    m1 = {"runs": [r]}
    m2 = {"runs": [r]}   # same run_id
    merged = merge_manifests([m1, m2])
    assert len(merged["runs"]) == 1


def test_merge_combines():
    h = {"key1": "abc"}
    r1 = _make_run("GPU_A", h)
    r2 = _make_run("GPU_B", h)
    merged = merge_manifests([{"runs": [r1]}, {"runs": [r2]}])
    assert len(merged["runs"]) == 2


def test_compare_agrees():
    h = {"key1": "abc123", "key2": "def456"}
    r1 = _make_run("GPU_A", h)
    r2 = _make_run("GPU_B", h)
    result = compare_manifests({"runs": [r1, r2]})
    assert result["agreed"] == 2
    assert result["failed"] == 0
    assert result["compliant"] is True


def test_compare_disagrees():
    r1 = _make_run("GPU_A", {"key1": "hash_a"})
    r2 = _make_run("GPU_B", {"key1": "hash_b"})
    result = compare_manifests({"runs": [r1, r2]})
    assert result["failed"] == 1
    assert result["compliant"] is False


def test_compare_same_platform_not_compliant():
    """Two runs on same GPU should agree but not count as cross-hardware proof."""
    h  = {"key1": "abc"}
    r1 = _make_run("GPU_A", h)
    r2 = {**r1, "run_id": "test_GPU_A_2"}
    result = compare_manifests({"runs": [r1, r2]})
    assert result["agreed"] == 1
    assert result["compliant"] is False   # same platform


def test_compare_needs_two_runs():
    m = build_manifest(force_cpu=True, gpu_name="TestCPU")
    result = compare_manifests(m)
    assert "error" in result


def test_hash_cross_cpu_determinism():
    """Two separate builds on CPU must produce identical hashes."""
    m1 = build_manifest(force_cpu=True, gpu_name="CPU_run1")
    m2 = build_manifest(force_cpu=True, gpu_name="CPU_run2")
    merged = merge_manifests([m1, m2])
    result = compare_manifests(merged)
    assert result["agreed"] == result["total"]
    assert result["failed"] == 0
