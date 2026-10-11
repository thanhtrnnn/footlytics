"""Downloads check disk and checksums; label extraction refuses old versions."""
import json
import zipfile
from types import SimpleNamespace

import pytest

from footlytics.bench import datasets, host
from footlytics.bench.paths import BenchPaths
from tests.bench_fixtures import make_labels


def make_zip(path, versions):
    with zipfile.ZipFile(path, "w") as z:
        for i, v in enumerate(versions):
            seq = f"SNGS-{990 + i}"
            z.writestr(f"{seq}/Labels-GameState.json", json.dumps(make_labels(seq=seq, version=v)))
            z.writestr(f"{seq}/img1/000001.jpg", b"\xff\xd8not-really")


def test_extract_labels_only(tmp_path):
    make_zip(tmp_path / "valid.zip", ["1.3", "1.3"])
    seqs = datasets.extract_gsr_labels(tmp_path / "valid.zip", tmp_path / "labels" / "valid")
    assert seqs == ["SNGS-990", "SNGS-991"]
    assert (tmp_path / "labels/valid/SNGS-990/Labels-GameState.json").exists()
    assert not list((tmp_path / "labels").rglob("*.jpg"))
    assert datasets.clips(BenchPaths(tmp_path / "root"), "valid") == []


def test_extract_refuses_old_labels(tmp_path):
    make_zip(tmp_path / "valid.zip", ["1.3", "1.2"])
    with pytest.raises(ValueError, match="1.3"):
        datasets.extract_gsr_labels(tmp_path / "valid.zip", tmp_path / "labels")


def test_fetch_hf_refuses_before_downloading_when_disk_is_short(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 5_000_000_000)
    info = SimpleNamespace(size=11_000_000_000, lfs=SimpleNamespace(sha256="x"))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])
    called = []
    with pytest.raises(host.ResourceError, match="11.0 GB"):
        datasets.fetch_hf("r", "valid.zip", "rev", tmp_path / "valid.zip", api=api,
                          download=lambda **k: called.append(k))
    assert called == []


def test_fetch_hf_deletes_a_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 50_000_000_000)
    info = SimpleNamespace(size=5, lfs=SimpleNamespace(sha256="0" * 64))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])

    def download(**k):
        p = tmp_path / "valid.zip"
        p.write_bytes(b"hello")
        return str(p)
    with pytest.raises(ValueError, match="sha256"):
        datasets.fetch_hf("r", "valid.zip", "rev", tmp_path / "valid.zip", api=api,
                          download=download)
    assert not (tmp_path / "valid.zip").exists()


def test_fetch_hf_skips_a_verified_file(tmp_path, monkeypatch):
    p = tmp_path / "valid.zip"
    p.write_bytes(b"hello")
    sha = datasets.sha256_file(p)
    (tmp_path / "valid.zip.sha256").write_text(sha)
    info = SimpleNamespace(size=5, lfs=SimpleNamespace(sha256=sha))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])
    assert datasets.fetch_hf("r", "valid.zip", "rev", p, api=api,
                             download=lambda **k: pytest.fail("re-downloaded")) == p


def test_label_extraction_checks_the_disk_first(tmp_path, monkeypatch):
    make_zip(tmp_path / "valid.zip", ["1.3", "1.3"])
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 1_500_000_100)
    with pytest.raises(host.ResourceError, match="free"):
        datasets.extract_gsr_labels(tmp_path / "valid.zip", tmp_path / "labels")
    assert not (tmp_path / "labels").exists()
