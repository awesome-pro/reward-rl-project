from __future__ import annotations

import json
from pathlib import Path

import pytest

from rewardforge.codec import decode_value
from rewardforge.config import Paths


TASKS = Paths().processed / "tasks_public.jsonl"
AUDIT = Paths().private / "audit_cases.jsonl"


pytestmark = pytest.mark.skipif(not TASKS.exists() or not AUDIT.exists(), reason="corpus not prepared")


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_public_tasks_do_not_leak_private_material():
    for row in _rows(TASKS):
        assert "audit_cases" not in row
        assert "checker_source" not in row
        assert "canonical_code" not in row
        for case in row["bank_cases"]:
            assert "expected" in case
            decode_value(case["args"])


def test_bank_and_audit_are_disjoint_by_task_and_source_index():
    tasks = {row["task_id"]: row for row in _rows(TASKS)}
    for row in _rows(AUDIT):
        task = tasks[row["task_id"]]
        bank = {case["source_index"] for case in task["bank_cases"]}
        audit = {case["source_index"] for case in row["audit_cases"]}
        assert bank.isdisjoint(audit)
        assert len(audit) >= 8
