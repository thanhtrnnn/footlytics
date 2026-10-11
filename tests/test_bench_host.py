"""Host checks: disk floor, GPU contention parsing, VRAM gate."""
import pytest

from footlytics.bench import host
from footlytics.bench.paths import BenchPaths


def test_paths_live_under_one_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTLYTICS_BENCH_ROOT", str(tmp_path))
    p = BenchPaths.from_env()
    assert p.root == tmp_path
    for q in (p.gsr_zip("valid"), p.gsr_labels, p.baseline_zip, p.run_dir("r1"), p.eval_python,
              p.trackeval, p.torch_pin):
        assert tmp_path in q.parents


def test_require_disk_refuses_below_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda path: 3_000_000_000)
    host.require_disk(tmp_path, 1_000_000_000)                 # 2 GB left > 1.5 GB floor
    with pytest.raises(host.ResourceError, match="free"):
        host.require_disk(tmp_path, 2_000_000_000)             # 1 GB left < floor


def test_free_disk_walks_up_to_an_existing_parent(tmp_path):
    assert host.free_disk_bytes(tmp_path / "not" / "yet") > 0


GPU = "2687, 24564\n"
APPS = "1661182, 2626\n4242, 900\n"


def test_parse_gpu_separates_other_users():
    r = host.parse_gpu(GPU, APPS, own_pids={4242})
    assert r == {"gpu_used_mib": 2687, "gpu_total_mib": 24564, "gpu_other_mib": 2626,
                 "gpu_procs": 2}


def test_require_vram_names_the_shortfall():
    readings = {"gpu_used_mib": 20000, "gpu_total_mib": 24564, "gpu_other_mib": 20000,
                "gpu_procs": 1}
    host.require_vram(4000, readings)
    with pytest.raises(host.ResourceError, match="VRAM"):
        host.require_vram(8000, readings)


def test_contention_without_nvidia_smi_still_reports_load(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(host.subprocess, "run", boom)
    r = host.contention()
    assert "load1" in r and r["gpu_used_mib"] is None


def test_calibration_paths_live_under_the_root(tmp_path):
    p = BenchPaths(tmp_path)
    for q in (p.weights, p.soccermaster_code, p.sn_calibration, p.calib_zip("valid")):
        assert tmp_path in q.parents
    assert p.calib_zip("test").name == "test.zip"
