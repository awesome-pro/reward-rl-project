"""P1 audit harness: base tests vs private enhanced-suite audit.

For every natural candidate:
  1. run the task's original/base tests (weak reward);
  2. if they pass, run the candidate-bank tests;
  3. independently run the private audit tests.

Only base-passing candidates are eligible for a positive reward, so only those
need bank/audit labels.  The distinction between candidate failure and harness
failure is recorded explicitly.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Paths, load_config
from .dataio import load_audit_cases, load_candidates, load_graders, load_tasks
from .sandbox import BaseResult, SandboxRunner, SuiteResult


def _outcome_dict(outcome: Any) -> dict[str, Any]:
    return {
        "index": outcome.index,
        "status": outcome.status,
        "failure_kind": outcome.failure_kind,
        "detail": outcome.detail,
        "wall_s": outcome.wall_s,
        "cpu_s": outcome.cpu_s,
        "source_index": outcome.source_index,
    }


def _suite_summary(result: SuiteResult) -> dict[str, Any]:
    passed = sum(1 for outcome in result.outcomes if outcome.status == "pass")
    return {
        "status": result.status,
        "all_passed": result.all_passed,
        "startup_s": result.startup_s,
        "startup_cpu_s": result.startup_cpu_s,
        "load_s": result.load_s,
        "backend": result.backend,
        "detail": result.detail,
        "tests_run": sum(1 for outcome in result.outcomes if outcome.status != "not_run"),
        "tests_passed": passed,
        "wall_s": sum(outcome.wall_s for outcome in result.outcomes),
        "cpu_s": sum(outcome.cpu_s for outcome in result.outcomes),
        "outcomes": [_outcome_dict(outcome) for outcome in result.outcomes],
    }


def _base_summary(result: BaseResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "pass": result.status == "pass",
        "wall_s": result.wall_s,
        "cpu_s": result.cpu_s,
        "detail": result.detail,
        "backend": result.backend,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cost_block(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "sum": sum(values) if values else None,
    }


def _load_existing(matrix_path: Path) -> set[str]:
    seen: set[str] = set()
    if not matrix_path.exists():
        return seen
    with open(matrix_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                seen.add(json.loads(line)["candidate_id"])
            except Exception:
                continue
    return seen


def run_audit(
    config_path: str | Path | None = None,
    *,
    limit: int | None = None,
    task_filter: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = Paths()
    artifacts = paths.artifacts / "audit"
    artifacts.mkdir(parents=True, exist_ok=True)
    matrix_path = artifacts / "matrix.jsonl"

    tasks = load_tasks(paths)
    audit_cases = load_audit_cases(paths)
    graders = load_graders(paths, tasks.keys())
    candidates = load_candidates(paths)
    if task_filter:
        candidates = [candidate for candidate in candidates if candidate["task_id"] == task_filter]
    candidates.sort(key=lambda candidate: (candidate["split"], candidate["task_id"], candidate["candidate_id"]))
    if limit is not None:
        candidates = candidates[:limit]

    seen = _load_existing(matrix_path) if resume else set()
    if not resume and matrix_path.exists():
        matrix_path.unlink()
    runner = SandboxRunner(config["sandbox"])
    mode = "a" if resume else "w"
    processed = 0
    started = time.monotonic()
    with open(matrix_path, mode, encoding="utf-8") as matrix_handle:
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen:
                continue
            task = tasks[candidate["task_id"]]
            checker = graders[candidate["task_id"]]
            entry_point = task["entry_point"]
            base = runner.run_base(candidate["code"], entry_point, task["base_test_code"])
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "task_id": candidate["task_id"],
                "benchmark": candidate["benchmark"],
                "split": candidate["split"],
                "generator": candidate["generator"],
                "family": candidate["family"],
                "code_sha256": candidate["code_sha256"],
                "base": _base_summary(base),
                "eligible_for_positive_reward": base.status == "pass",
                "excluded": base.status == "infra_error",
                "exclusion_reason": "base_infra_error" if base.status == "infra_error" else "",
                "bank": None,
                "audit": None,
                "audited_correct": None,
                "gate_relevant": False,
            }
            if base.status != "infra_error":
                bank = runner.run_suite(
                    candidate["code"], entry_point, task["bank_cases"], checker, stop_on_failure=False
                )
                audit = runner.run_suite(
                    candidate["code"], entry_point, audit_cases[candidate["task_id"]], checker, stop_on_failure=False
                )
                record["bank"] = _suite_summary(bank)
                record["audit"] = _suite_summary(audit)
                infra = bank.status == "infra_error" or audit.status == "infra_error"
                if infra:
                    record["excluded"] = True
                    record["exclusion_reason"] = "suite_infra_error"
                record["audited_correct"] = audit.all_passed if not infra else None
                # Only base-passing, successfully audited candidates are
                # eligible for the policy/FPR comparison.  Base-fail labels are
                # still useful for measuring base-test false rejection.
                record["gate_relevant"] = base.status == "pass" and not infra
            matrix_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            matrix_handle.flush()
            processed += 1
            if processed % 25 == 0:
                elapsed = time.monotonic() - started
                print(f"audited {processed}/{len(candidates)} candidates ({elapsed:.1f}s)", flush=True)

    summary = summarize_audit(paths, config, matrix_path, runner.backend)
    return summary


def summarize_audit(
    paths: Paths,
    config: dict[str, Any],
    matrix_path: Path,
    backend: str,
) -> dict[str, Any]:
    records = []
    with open(matrix_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    artifacts = matrix_path.parent

    base_pass = [record for record in records if record["eligible_for_positive_reward"]]
    audited = [record for record in base_pass if record.get("audited_correct") is not None and not record["excluded"]]
    base_pass_audit_fail = [record for record in audited if not record["audited_correct"]]
    base_pass_audit_pass = [record for record in audited if record["audited_correct"]]
    all_audited = [
        record for record in records
        if record.get("audited_correct") is not None and not record["excluded"]
    ]
    base_fail_audited = [record for record in all_audited if not record["eligible_for_positive_reward"]]
    base_fail_audit_pass = [record for record in base_fail_audited if record["audited_correct"]]
    base_fail_audit_pass_rate = (
        len(base_fail_audit_pass) / (len(base_fail_audit_pass) + len(base_pass_audit_pass))
        if (base_fail_audit_pass or base_pass_audit_pass)
        else None
    )

    def breakdown(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in audited:
            groups[str(record.get(key))].append(record)
        out: dict[str, Any] = {}
        for name, rows in sorted(groups.items()):
            fails = sum(1 for row in rows if not row["audited_correct"])
            out[name] = {
                "n": len(rows),
                "base_pass_audit_fail": fails,
                "base_pass_audit_pass": len(rows) - fails,
                "rate": fails / len(rows) if rows else None,
            }
        return out

    def cost_values(rows: list[dict[str, Any]], suite: str) -> tuple[list[float], list[float], list[float]]:
        wall = [row[suite]["wall_s"] + row[suite]["startup_s"] for row in rows if row.get(suite)]
        cpu = [row[suite]["cpu_s"] + row[suite]["startup_cpu_s"] for row in rows if row.get(suite)]
        startup = [row[suite]["startup_s"] for row in rows if row.get(suite)]
        return wall, cpu, startup

    bank_wall, bank_cpu, bank_startup = cost_values(audited, "bank")
    audit_wall, audit_cpu, _ = cost_values(audited, "audit")
    base_wall = [row["base"]["wall_s"] for row in audited]
    base_cpu = [row["base"]["cpu_s"] for row in audited]
    test_wall = [
        outcome["wall_s"]
        for row in audited
        if row.get("bank")
        for outcome in row["bank"]["outcomes"]
        if outcome["status"] != "not_run"
    ]
    candidate_test_wall = [
        sum(outcome["wall_s"] for outcome in row["bank"]["outcomes"] if outcome["status"] != "not_run")
        for row in audited
        if row.get("bank")
    ]

    eval_audited = [record for record in audited if record["split"] == "eval"]
    eval_fail = [record for record in eval_audited if not record["audited_correct"]]
    eval_fail_tasks = sorted({record["task_id"] for record in eval_fail})

    def gate_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
        fails = [row for row in rows if not row["audited_correct"]]
        return {
            "base_pass_audited": len(rows),
            "base_pass_audit_fail": len(fails),
            "base_pass_audit_pass": len(rows) - len(fails),
            "distinct_tasks_with_audit_fail": len({row["task_id"] for row in fails}),
            "distinct_generators_with_audit_fail": len({row["generator"] for row in fails}),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "first experiment / P1 audit",
        "sandbox_backend": backend,
        "counts": {
            "candidates_in_matrix": len(records),
            "excluded_infra": sum(1 for record in records if record["excluded"]),
            "base_pass": len(base_pass),
            "base_fail": sum(1 for record in records if not record["eligible_for_positive_reward"]),
            "base_pass_audited": len(audited),
            "base_pass_audit_fail": len(base_pass_audit_fail),
            "base_pass_audit_pass": len(base_pass_audit_pass),
            "all_audited": len(all_audited),
            "base_fail_audited": len(base_fail_audited),
            "base_fail_audit_pass": len(base_fail_audit_pass),
            "base_fail_audit_pass_rate": base_fail_audit_pass_rate,
            "base_fail_audit_pass_note": (
                "Audit cases exclude base-overlap inputs; passing the sampled audit does not prove "
                "the base test was wrong. Treat this as an upper bound on base-test false rejection."
            ),
        },
        "rates": {
            "base_pass_rate": len(base_pass) / len(records) if records else None,
            "base_pass_audit_fail_rate": len(base_pass_audit_fail) / len(audited) if audited else None,
            "base_fail_audit_pass_rate": base_fail_audit_pass_rate,
        },
        "by_split": {},
        "by_benchmark": breakdown("benchmark"),
        "by_generator": breakdown("generator"),
        "by_family": breakdown("family"),
        "gate_1_eval": gate_counts(eval_audited),
        "eval_audit_fail_tasks": eval_fail_tasks,
        "costs": {
            "base_wall_s": _cost_block(base_wall),
            "base_cpu_s": _cost_block(base_cpu),
            "bank_total_wall_s": _cost_block(bank_wall),
            "bank_total_cpu_s": _cost_block(bank_cpu),
            "bank_startup_wall_s": _cost_block(bank_startup),
            "bank_test_wall_s": _cost_block(test_wall),
            "audit_total_wall_s": _cost_block(audit_wall),
            "audit_total_cpu_s": _cost_block(audit_cpu),
        },
        "bank_test_executions": sum(
            row["bank"]["tests_run"] for row in audited if row.get("bank")
        ),
        "audit_test_executions": sum(
            row["audit"]["tests_run"] for row in audited if row.get("audit")
        ),
        "gate_1": {
            "criterion": ">=30 naturally generated base-pass/audit-fail candidates on held-out eval, multiple tasks",
            "observed": len(eval_fail),
            "distinct_tasks": len(eval_fail_tasks),
            "threshold": 30,
            "passed": len(eval_fail) >= 30 and len(eval_fail_tasks) >= 5,
        },
        "gate_5": {
            "criterion": "real overhead high enough that exhaustive checking is not trivially cheap",
            "full_bank_p50_s": _percentile(bank_wall, 0.50),
            "base_p50_s": _percentile(base_wall, 0.50),
            "median_per_test_wall_s": _percentile(test_wall, 0.50),
            "median_candidate_test_total_s": _percentile(candidate_test_wall, 0.50),
            "median_bank_startup_s": _percentile(bank_startup, 0.50),
            "median_startup_share_of_bank": (
                (_percentile(bank_startup, 0.50) or 0.0) / (_percentile(bank_wall, 0.50) or 1.0)
            ),
            "median_test_fraction_of_total": (
                (_percentile(candidate_test_wall, 0.50) or 0.0)
                / ((_percentile(bank_wall, 0.50) or 0.0) + (_percentile(base_wall, 0.50) or 0.0) or 1.0)
            ),
            "startup_share_of_bank_mean": (
                statistics.fmean(bank_startup) / statistics.fmean(bank_wall)
                if bank_wall and statistics.fmean(bank_wall) > 0
                else None
            ),
            "note": "Median is the decision-relevant figure; the mean is inflated by slow-tail candidates.",
        },
        "dataset": {
            "manifest": "data/manifest.json",
            "policy_input": "data/processed/tasks_public.jsonl",
            "private_audit_input": "data/private/audit_cases.jsonl",
        },
    }
    for split in ("dev", "eval"):
        rows = [record for record in audited if record["split"] == split]
        summary["by_split"][split] = gate_counts(rows)

    # Full case-level table for independent analysis.
    cases_path = artifacts / "cases.csv"
    with open(cases_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "task_id",
                "benchmark",
                "split",
                "generator",
                "family",
                "suite",
                "case_index",
                "source_index",
                "status",
                "failure_kind",
                "wall_s",
                "cpu_s",
            ]
        )
        for record in records:
            for suite_name in ("bank", "audit"):
                suite = record.get(suite_name)
                if not suite:
                    continue
                for outcome in suite["outcomes"]:
                    writer.writerow(
                        [
                            record["candidate_id"],
                            record["task_id"],
                            record["benchmark"],
                            record["split"],
                            record["generator"],
                            record["family"],
                            suite_name,
                            outcome["index"],
                            outcome.get("source_index", ""),
                            outcome["status"],
                            outcome["failure_kind"],
                            outcome["wall_s"],
                            outcome["cpu_s"],
                        ]
                    )

    candidates_path = artifacts / "candidates.csv"
    with open(candidates_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "candidate_id",
                "task_id",
                "benchmark",
                "split",
                "generator",
                "family",
                "base_status",
                "base_pass",
                "audited_correct",
                "base_wall_s",
                "bank_total_wall_s",
                "bank_startup_s",
                "audit_total_wall_s",
                "bank_first_failure",
                "audit_first_failure",
                "excluded",
                "exclusion_reason",
            ]
        )
        for record in records:
            bank = record.get("bank")
            audit = record.get("audit")
            bank_first = ""
            if bank:
                failure = next((o for o in bank["outcomes"] if o["status"] != "pass"), None)
                bank_first = failure["failure_kind"] if failure else ""
            audit_first = ""
            if audit:
                failure = next((o for o in audit["outcomes"] if o["status"] != "pass"), None)
                audit_first = failure["failure_kind"] if failure else ""
            writer.writerow(
                [
                    record["candidate_id"],
                    record["task_id"],
                    record["benchmark"],
                    record["split"],
                    record["generator"],
                    record["family"],
                    record["base"]["status"],
                    record["eligible_for_positive_reward"],
                    record["audited_correct"],
                    record["base"]["wall_s"],
                    (bank["wall_s"] + bank["startup_s"]) if bank else "",
                    bank["startup_s"] if bank else "",
                    (audit["wall_s"] + audit["startup_s"]) if audit else "",
                    bank_first,
                    audit_first,
                    record["excluded"],
                    record["exclusion_reason"],
                ]
            )

    summary_path = artifacts / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Run the RewardForge P1 audit")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    summary = run_audit(args.config, limit=args.limit, task_filter=args.task, resume=not args.no_resume)
    print(json.dumps({k: summary[k] for k in ("counts", "rates", "gate_1", "gate_5")}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
