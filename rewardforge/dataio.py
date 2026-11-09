"""Loaders for the frozen corpus.

``load_tasks`` reads only the public task file (bank tests).  Private audit
cases and graders are loaded explicitly by the audit harness.  This module is
the enforcement point for the policy's data separation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .codec import decode_value
from .config import Paths
from .testspec import load_checker


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decode_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "args": decode_value(case["args"]),
        "expected": decode_value(case["expected"]),
    }


def load_tasks(paths: Paths | None = None, split: str | None = None) -> dict[str, dict[str, Any]]:
    paths = paths or Paths()
    rows = read_jsonl(paths.processed / "tasks_public.jsonl")
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        row = dict(row)
        row["bank_cases"] = [decode_case(case) for case in row.get("bank_cases", [])]
        tasks[row["task_id"]] = row
    return tasks


def load_audit_cases(paths: Paths | None = None) -> dict[str, list[dict[str, Any]]]:
    paths = paths or Paths()
    rows = read_jsonl(paths.private / "audit_cases.jsonl")
    return {row["task_id"]: [decode_case(case) for case in row["audit_cases"]] for row in rows}


def load_graders(paths: Paths | None = None, task_ids: Iterable[str] | None = None) -> dict[str, Any]:
    paths = paths or Paths()
    keep = set(task_ids) if task_ids is not None else None
    rows = read_jsonl(paths.private / "graders.jsonl")
    graders: dict[str, Any] = {}
    for row in rows:
        if keep is not None and row["task_id"] not in keep:
            continue
        graders[row["task_id"]] = load_checker(row["checker_source"])
    return graders


def load_candidates(
    paths: Paths | None = None, split: str | None = None
) -> list[dict[str, Any]]:
    paths = paths or Paths()
    rows = read_jsonl(paths.processed / "candidates.jsonl")
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows


def load_splits(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or Paths()
    with open(paths.processed / "splits.json", encoding="utf-8") as handle:
        return json.load(handle)
