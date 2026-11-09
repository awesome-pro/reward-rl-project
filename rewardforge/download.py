"""Pinned downloads for the RewardForge pilot.

Every artifact is content-addressed in ``data/manifest.json``.  Re-running
``rewardforge prepare`` verifies cached files against the pinned SHA-256 and
only downloads what is missing or corrupt.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Paths, load_config


EVALPLUS_RELEASE_BASE = "https://github.com/evalplus/evalplus/releases/download"

# EvalPlus dataset revisions (HuggingFace) pin the exact bytes.
HUMANEVALPLUS_REV = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
MBPPPLUS_REV = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
HUMANEVAL_REV = "b9453e8f1865df0a0057a1383767caa400acfbcf"

DATASETS: dict[str, dict[str, str]] = {
    "humanevalplus": {
        "url": f"https://huggingface.co/datasets/evalplus/humanevalplus/resolve/{HUMANEVALPLUS_REV}/test.jsonl",
        "sha256": "908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492",
        "path": "humanevalplus.jsonl",
        "revision": HUMANEVALPLUS_REV,
    },
    "mbppplus": {
        "url": f"https://huggingface.co/datasets/evalplus/mbppplus/resolve/{MBPPPLUS_REV}/data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "sha256": "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
        "path": "mbppplus.parquet",
        "revision": MBPPPLUS_REV,
    },
    "humaneval": {
        "url": f"https://raw.githubusercontent.com/openai/human-eval/{HUMANEVAL_REV}/data/HumanEval.jsonl.gz",
        "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        "path": "humaneval.jsonl.gz",
        "revision": HUMANEVAL_REV,
    },
}

# Release tags and asset hashes for the natural candidate corpus.
CANDIDATE_ASSETS: dict[tuple[str, str], dict[str, str]] = {
    ("humaneval", "gpt-4-1106-preview_temp_0.0.zip"): {
        "release": "v0.1.0",
        "sha256": "cad51c9c1d1f0491adc79cbb5739e10891472497e391d510cbd27ed9707b6015",
    },
    ("humaneval", "chatgpt_temp_0.0.zip"): {
        "release": "v0.1.0",
        "sha256": "456a49813ad3afac920b92e53d921cf0a2693f06cb980a85a2ef2e55a717dd47",
    },
    ("humaneval", "codegen-2b_temp_0.0.zip"): {
        "release": "v0.1.0",
        "sha256": "9e5a3c8327401098e8d85d1a079bc1d9aed76bef8edefdd704ee14782109bf4f",
    },
    ("mbpp", "gpt-4-1106-preview_temp_0.0.zip"): {
        "release": "v0.2.0",
        "sha256": "4e060bf3473b09b763567e042c693fd99d4922d9602124dcb70ab184bcbb252d",
    },
    ("mbpp", "deepseek-coder-6.7b-base_temp_0.zip"): {
        "release": "v0.2.0",
        "sha256": "ec8036508d353c23884d290d58b38672964af8584dd82a4fffa9f78cce4101fb",
    },
    ("mbpp", "codegen-2b_temp_0.zip"): {
        "release": "v0.2.0",
        "sha256": "6b1826536782f102f671bf5cc270e51fd61c10c66fb7708b1085e1d69300208a",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rewardforge/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, open(temporary, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(destination)


def ensure_dataset(name: str, paths: Paths, *, force: bool = False) -> dict[str, str]:
    spec = DATASETS[name]
    destination = paths.raw / spec["path"]
    if force or not destination.exists() or sha256_file(destination) != spec["sha256"]:
        _download(spec["url"], destination)
    actual = sha256_file(destination)
    if actual != spec["sha256"]:
        raise RuntimeError(f"hash mismatch for {name}: {actual} != {spec['sha256']}")
    return {
        "name": name,
        "url": spec["url"],
        "revision": spec["revision"],
        "path": str(destination.relative_to(paths.root)),
        "sha256": actual,
        "size": destination.stat().st_size,
    }


def ensure_candidate_assets(
    config: dict[str, Any], paths: Paths, *, force: bool = False
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    extracted: list[dict[str, str]] = []
    for benchmark, generators in config["generators"].items():
        for generator in generators:
            asset = generator["asset"]
            key = (benchmark, asset)
            spec = CANDIDATE_ASSETS.get(key)
            if spec is None:
                raise RuntimeError(f"no pinned hash for candidate asset {key}")
            destination = paths.raw / "candidates" / benchmark / asset
            if force or not destination.exists() or sha256_file(destination) != spec["sha256"]:
                url = f"{EVALPLUS_RELEASE_BASE}/{spec['release']}/{asset}"
                _download(url, destination)
            actual = sha256_file(destination)
            if actual != spec["sha256"]:
                raise RuntimeError(f"hash mismatch for {asset}: {actual} != {spec['sha256']}")
            extract_dir = destination.with_suffix("")
            marker = extract_dir / ".extracted"
            if force or not marker.exists():
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(destination) as archive:
                    archive.extractall(extract_dir)
                marker.write_text(actual, encoding="utf-8")
            records.append(
                {
                    "benchmark": benchmark,
                    "generator": generator["name"],
                    "family": generator["family"],
                    "release": spec["release"],
                    "asset": asset,
                    "url": f"{EVALPLUS_RELEASE_BASE}/{spec['release']}/{asset}",
                    "sha256": actual,
                    "size": destination.stat().st_size,
                    "extracted_dir": str(extract_dir.relative_to(paths.root)),
                }
            )
            extracted.append(
                {
                    "benchmark": benchmark,
                    "generator": generator["name"],
                    "family": generator["family"],
                    "dir": str(extract_dir),
                }
            )
    return records, extracted


def download_everything(
    config_path: str | Path | None = None, *, force: bool = False
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = Paths()
    paths.ensure()
    datasets = [ensure_dataset(name, paths, force=force) for name in ("humanevalplus", "mbppplus", "humaneval")]
    assets, extracted = ensure_candidate_assets(config, paths, force=force)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evalplus_release": "v0.1.0 (HumanEval candidates), v0.2.0 (MBPP candidates)",
        "datasets": datasets,
        "candidate_assets": assets,
        "notes": [
            "HumanEval+ and MBPP+ are used as the audited benchmark, not as a claim of new test quality.",
            "Base tests are the task's weak-reward tests: original HumanEval check() for HumanEval, and the EvalPlus base test_list for MBPP+.",
            "Candidate programs are public, previously generated model outputs from the pinned EvalPlus releases.",
        ],
    }
    manifest_path = paths.raw / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "extracted": extracted}
