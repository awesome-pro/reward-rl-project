from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.json"


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    private: Path = ROOT / "data" / "private"
    artifacts: Path = ROOT / "artifacts"

    def ensure(self) -> None:
        for path in (self.raw, self.processed, self.private, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def stable_seed(*parts: object) -> int:
    """Stable across processes (unlike built-in hash())."""
    import hashlib

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_key(*parts: object) -> str:
    import hashlib

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
