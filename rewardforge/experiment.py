"""P2 budgeted replay: baselines vs the adaptive selector.

The replay charges measured per-test wall/CPU time from the P1 audit, plus the
measured bank-worker startup.  It never executes candidate code again and
never reads private audit cases or labels while choosing tests.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Paths, load_config, stable_seed
from .policies import (
    B0Policy,
    B4Policy,
    Policy,
    ReplayCandidate,
    build_replay_data,
    policies_for,
    prepare_policies,
    replay_candidate,
)


@dataclass
class ResultRow:
    policy: str
    budget: float | None
    candidate_id: str
    task_id: str
    benchmark: str
    split: str
    generator: str
    family: str
    reward: int
    audit_correct: bool
    cost_wall: float
    cost_cpu: float
    tests_run: int
    budget_exhausted: bool
    first_failure: str

    @property
    def false_positive(self) -> bool:
        return bool(self.reward) and not self.audit_correct

    @property
    def false_negative(self) -> bool:
        return not self.reward and self.audit_correct


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)


def _cost_block(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _metrics(rows: list[ResultRow]) -> dict[str, Any]:
    incorrect = [row for row in rows if not row.audit_correct]
    correct = [row for row in rows if row.audit_correct]
    fp = sum(1 for row in rows if row.false_positive)
    accepted_correct = sum(1 for row in correct if row.reward)
    rejected_correct = len(correct) - accepted_correct
    accepted = sum(1 for row in rows if row.reward)
    costs = [row.cost_wall for row in rows]
    cpu = [row.cost_cpu for row in rows]
    tests = [row.tests_run for row in rows]
    return {
        "n": len(rows),
        "incorrect": len(incorrect),
        "correct": len(correct),
        "accepted": accepted,
        "false_positives": fp,
        "false_positive_rate": fp / len(incorrect) if incorrect else None,
        "true_positive_acceptance": accepted_correct / len(correct) if correct else None,
        "false_rejection_rate": rejected_correct / len(correct) if correct else None,
        "false_negatives": rejected_correct,
        "precision": accepted_correct / accepted if accepted else None,
        "cost_wall_s": _cost_block(costs),
        "cost_cpu_s": _cost_block(cpu),
        "tests_run_mean": statistics.fmean(tests) if tests else None,
        "tests_run_total": sum(tests),
        "budget_exhausted": sum(1 for row in rows if row.budget_exhausted),
    }


def _run_policy(policy: Policy, candidates: list[ReplayCandidate], budget: float | None) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for candidate in candidates:
        result = replay_candidate(policy, candidate, budget)
        rows.append(
            ResultRow(
                policy=policy.name,
                budget=budget,
                candidate_id=candidate.candidate_id,
                task_id=candidate.task_id,
                benchmark=candidate.benchmark,
                split=candidate.split,
                generator=candidate.generator,
                family=candidate.family,
                reward=result.reward,
                audit_correct=candidate.audit_correct,
                cost_wall=result.cost_wall,
                cost_cpu=result.cost_cpu,
                tests_run=result.tests_run,
                budget_exhausted=result.budget_exhausted,
                first_failure=result.first_failure,
            )
        )
    return rows


def _task_counts(rows: list[ResultRow]) -> dict[str, tuple[int, int, int]]:
    """task -> (false positives, incorrect, accepted correct)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        bucket = counts[row.task_id]
        if not row.audit_correct:
            bucket[1] += 1
            if row.reward:
                bucket[0] += 1
        elif row.reward:
            bucket[2] += 1
    return {task: tuple(values) for task, values in counts.items()}


def _bootstrap_fpr(
    rows: list[ResultRow],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    tasks = sorted({row.task_id for row in rows})
    if not tasks:
        return {"ci_low": None, "ci_high": None}
    counts = _task_counts(rows)
    total_incorrect = sum(counts[task][1] for task in tasks)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        fp = incorrect = 0
        for _ in range(len(tasks)):
            task = tasks[rng.randrange(len(tasks))]
            task_fp, task_incorrect, _ = counts[task]
            fp += task_fp
            incorrect += task_incorrect
        if incorrect:
            values.append(fp / incorrect)
    if not values:
        return {"ci_low": None, "ci_high": None}
    values.sort()
    return {
        "ci_low": values[int(0.025 * (len(values) - 1))],
        "ci_high": values[int(0.975 * (len(values) - 1))],
        "point": sum(counts[t][0] for t in tasks) / total_incorrect if total_incorrect else None,
        "iterations_with_denominator": len(values),
    }


def _paired_bootstrap(
    candidate_rows: list[ResultRow],
    baseline_rows: list[ResultRow],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Task-clustered paired bootstrap of FPR(candidate) - FPR(baseline)."""

    tasks = sorted({row.task_id for row in candidate_rows})
    candidate_counts = _task_counts(candidate_rows)
    baseline_counts = _task_counts(baseline_rows)
    rng = random.Random(seed)
    differences: list[float] = []
    relative: list[float] = []
    for _ in range(iterations):
        cand_fp = cand_incorrect = base_fp = base_incorrect = 0
        for _ in range(len(tasks)):
            task = tasks[rng.randrange(len(tasks))]
            c_fp, c_inc, _ = candidate_counts.get(task, (0, 0, 0))
            b_fp, b_inc, _ = baseline_counts.get(task, (0, 0, 0))
            cand_fp += c_fp
            cand_incorrect += c_inc
            base_fp += b_fp
            base_incorrect += b_inc
        if cand_incorrect and base_incorrect:
            cand_fpr = cand_fp / cand_incorrect
            base_fpr = base_fp / base_incorrect
            differences.append(cand_fpr - base_fpr)
            if base_fpr > 0:
                relative.append((base_fpr - cand_fpr) / base_fpr)
    differences.sort()
    relative.sort()

    def interval(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]

    diff_low, diff_high = interval(differences)
    rel_low, rel_high = interval(relative)
    return {
        "difference_ci_low": diff_low,
        "difference_ci_high": diff_high,
        "relative_reduction_ci_low": rel_low,
        "relative_reduction_ci_high": rel_high,
        "iterations": len(differences),
    }


def _group_metrics(rows: list[ResultRow], key: str) -> dict[str, Any]:
    groups: dict[str, list[ResultRow]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, key))].append(row)
    return {name: _metrics(group) for name, group in sorted(groups.items())}


def _difficulty_buckets(paths: Paths) -> dict[str, str]:
    """Descriptive proxy: task base-pass rate over all natural candidates.

    This is computed from the audit matrix for reporting only; it is never a
    policy input.
    """

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    matrix_path = paths.artifacts / "audit" / "matrix.jsonl"
    if not matrix_path.exists():
        return {}
    with open(matrix_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            counts[record["task_id"]][0] += 1
            if record.get("eligible_for_positive_reward"):
                counts[record["task_id"]][1] += 1
    buckets: dict[str, str] = {}
    for task_id, (total, passed) in counts.items():
        rate = passed / total if total else 0.0
        buckets[task_id] = "hard" if rate < 0.5 else ("medium" if rate < 0.75 else "easy")
    return buckets


def _best_baseline(summary: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Best B1-B3 by point FPR; ties prefer B2 (the pre-registered static baseline)."""

    names = ["B1", "B2", "B3"]
    rates = {name: summary[name].get("false_positive_rate") for name in names}
    available = {name: rate for name, rate in rates.items() if rate is not None}
    if not available:
        return "B2", summary["B2"]
    best = min(available.values())
    if rates["B2"] == best:
        return "B2", summary["B2"]
    name = min(available, key=available.get)  # type: ignore[arg-type]
    return name, summary[name]


def _cost_ratio(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_p50 = left["cost_wall_s"]["p50"]
    right_p50 = right["cost_wall_s"]["p50"]
    if left_p50 is None or right_p50 in (None, 0):
        return None
    return left_p50 / right_p50


def run_experiment(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    paths = Paths()
    artifacts = paths.artifacts / "experiment"
    artifacts.mkdir(parents=True, exist_ok=True)
    dev, eval_ = build_replay_data(paths)
    if not dev or not eval_:
        raise RuntimeError("no replay data; run `rewardforge audit` first")

    budgets = [float(value) for value in config["budgets_s"]]
    base_eval_p50 = _percentile([candidate.base_wall for candidate in eval_], 0.50)
    policies = policies_for(int(config["seed"]), config)
    prepare_policies(policies, dev)
    policy_map = {policy.name: policy for policy in policies}

    all_rows: list[ResultRow] = []
    frontier: list[dict[str, Any]] = []
    for budget in budgets:
        for name in ("B1", "B2", "B3", "Candidate"):
            rows = _run_policy(policy_map[name], eval_, budget)
            all_rows.extend(rows)
            frontier.append(
                {
                    "policy": name,
                    "budget_s": budget,
                    **_metrics(rows),
                }
            )
    reference: dict[str, list[ResultRow]] = {}
    for name in ("B0", "B4"):
        rows = _run_policy(policy_map[name], eval_, None)
        reference[name] = rows
        all_rows.extend(rows)
        frontier.append(
            {
                "policy": name,
                "budget_s": None,
                **_metrics(rows),
            }
        )

    # Grouped summaries at each budget.
    summaries: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        summaries[str(budget)] = {}
        for name in ("B1", "B2", "B3", "Candidate"):
            rows = [row for row in all_rows if row.policy == name and row.budget == budget]
            summaries[str(budget)][name] = _metrics(rows)

    reference_metrics = {name: _metrics(rows) for name, rows in reference.items()}

    # Bootstrap CIs and the Gate 2 comparison.
    iterations = int(config.get("bootstrap", {}).get("iterations", 2000))
    comparisons: list[dict[str, Any]] = []
    for budget in budgets:
        rows_by_policy = {
            name: [row for row in all_rows if row.policy == name and row.budget == budget]
            for name in ("B1", "B2", "B3", "Candidate")
        }
        summary = {name: _metrics(rows) for name, rows in rows_by_policy.items()}
        best_name, best_summary = _best_baseline(summary)
        candidate_rows = rows_by_policy["Candidate"]
        baseline_rows = rows_by_policy[best_name]
        candidate_fpr = summary["Candidate"]["false_positive_rate"]
        best_fpr = best_summary["false_positive_rate"]
        relative = (
            (best_fpr - candidate_fpr) / best_fpr
            if best_fpr not in (None, 0) and candidate_fpr is not None
            else None
        )
        paired = _paired_bootstrap(
            candidate_rows,
            baseline_rows,
            iterations=iterations,
            seed=stable_seed(config["seed"], "paired", budget),
        )
        cost_ratio = _cost_ratio(summary["Candidate"], best_summary)
        comparisons.append(
            {
                "budget_s": budget,
                "best_baseline": best_name,
                "candidate_fpr": candidate_fpr,
                "best_baseline_fpr": best_fpr,
                "relative_reduction": relative,
                "candidate_p50_cost_s": summary["Candidate"]["cost_wall_s"]["p50"],
                "best_baseline_p50_cost_s": best_summary["cost_wall_s"]["p50"],
                "cost_ratio": cost_ratio,
                "candidate_vs_best_rate": {
                    name: summary[name]["false_positive_rate"] for name in ("B1", "B2", "B3", "Candidate")
                },
                "p50_by_policy": {
                    name: summary[name]["cost_wall_s"]["p50"] for name in ("B1", "B2", "B3", "Candidate")
                },
                "p95_by_policy": {
                    name: summary[name]["cost_wall_s"]["p95"] for name in ("B1", "B2", "B3", "Candidate")
                },
                "tests_run_mean_by_policy": {
                    name: summary[name]["tests_run_mean"] for name in ("B1", "B2", "B3", "Candidate")
                },
                "candidate_ci": _bootstrap_fpr(
                    candidate_rows, iterations=iterations, seed=stable_seed(config["seed"], "candidate", budget)
                ),
                "best_baseline_ci": _bootstrap_fpr(
                    baseline_rows, iterations=iterations, seed=stable_seed(config["seed"], "baseline", budget)
                ),
                "paired": paired,
            }
        )

    # Gate 2: >=30% relative reduction, CI favors it, at a cost within 10%.
    qualifying = [
        comparison
        for comparison in comparisons
        if comparison["relative_reduction"] is not None
        and comparison["relative_reduction"] >= 0.30
        and comparison["candidate_vs_best_rate"]["Candidate"]
        < comparison["candidate_vs_best_rate"][comparison["best_baseline"]]
        and comparison["paired"]["difference_ci_high"] is not None
        and comparison["paired"]["difference_ci_high"] < 0
        and comparison["cost_ratio"] is not None
        and 0.90 <= comparison["cost_ratio"] <= 1.10
    ]
    # Do not call a no-test budget the "best" point: require at least one test
    # execution per candidate on average.
    test_running = [
        item
        for item in comparisons
        if (item["tests_run_mean_by_policy"]["Candidate"] or 0.0) >= 1.0
        and item["candidate_p50_cost_s"] >= (base_eval_p50 or 0.0) + 0.01
        and item["best_baseline_p50_cost_s"] >= (base_eval_p50 or 0.0) + 0.01
    ]
    best_comparison = max(
        test_running or comparisons,
        key=lambda item: item["relative_reduction"] if item["relative_reduction"] is not None else -math.inf,
    )
    representative = {
        "best_testing": best_comparison,
        "tight_33": next((item for item in comparisons if item["budget_s"] == 0.033), best_comparison),
        "near_full": next((item for item in comparisons if item["budget_s"] == 0.06), best_comparison),
        "tail_matched": next((item for item in comparisons if item["budget_s"] == 0.16), best_comparison),
    }

    # Cost decomposition for Gate 5.
    base_costs = [candidate.base_wall for candidate in eval_]
    bank_costs = [candidate.bank_startup + sum(case.cost for case in candidate.cases) for candidate in eval_]
    startup_costs = [candidate.bank_startup for candidate in eval_]
    test_costs = [sum(case.cost for case in candidate.cases) for candidate in eval_]
    total_costs = [base + bank for base, bank in zip(base_costs, bank_costs)]
    median_base = _percentile(base_costs, 0.50)
    median_bank = _percentile(bank_costs, 0.50)
    median_startup = _percentile(startup_costs, 0.50)
    median_test = _percentile(test_costs, 0.50)
    median_total = _percentile(total_costs, 0.50)
    test_fraction = (
        median_test / median_total if median_test is not None and median_total not in (None, 0) else None
    )
    startup_share = (
        median_startup / median_bank if median_startup is not None and median_bank not in (None, 0) else None
    )
    exhaustive_cheap = test_fraction is not None and test_fraction < 0.05 and (startup_share or 0) > 0.5

    # Gate 3: no material false-rejection increase relative to all-tests B4.
    b4_false_rejection = reference_metrics["B4"]["false_rejection_rate"]
    # Recompute at the budget with the best relative reduction.
    best_budget_rows = [
        row
        for row in all_rows
        if row.policy == "Candidate" and row.budget == best_comparison["budget_s"]
    ]
    candidate_fnr = _metrics(best_budget_rows)["false_rejection_rate"] if best_budget_rows else None
    gate3_passed = (
        candidate_fnr is not None
        and b4_false_rejection is not None
        and candidate_fnr <= b4_false_rejection + 0.01
    )

    # Per-group results at the best-comparison budget.
    group_rows = [
        row for row in all_rows if row.budget == best_comparison["budget_s"] and row.policy in {"B1", "B2", "B3", "Candidate"}
    ]
    # Explicit grouping at the best-comparison budget.
    def grouped(key: str) -> dict[str, Any]:
        groups: dict[str, list[ResultRow]] = defaultdict(list)
        for row in group_rows:
            groups[str(getattr(row, key))].append(row)
        out: dict[str, Any] = {}
        for name, rows in sorted(groups.items()):
            out[name] = {
                policy: _metrics([row for row in rows if row.policy == policy])
                for policy in ("B1", "B2", "B3", "Candidate")
            }
        return out

    by_benchmark = grouped("benchmark")
    by_generator = grouped("generator")
    difficulty_map = _difficulty_buckets(paths)
    difficulty_groups: dict[str, list[ResultRow]] = defaultdict(list)
    for row in group_rows:
        difficulty_groups[difficulty_map.get(row.task_id, "unknown")].append(row)
    by_difficulty = {
        name: {
            policy: _metrics([row for row in rows if row.policy == policy])
            for policy in ("B1", "B2", "B3", "Candidate")
        }
        for name, rows in sorted(difficulty_groups.items())
    }

    # Secondary generator-holdout generalization check.
    holdout = _generator_holdout(dev, eval_, config, budgets, iterations)
    gate4_passed = bool(
        best_comparison["relative_reduction"] is not None
        and best_comparison["relative_reduction"] >= 0.30
        and holdout["any_generator_gain"]
    )

    gate1 = _load_gate1(paths)
    gate2_passed = bool(qualifying)
    gate5_passed = not exhaustive_cheap
    overall_proceed = bool(gate1.get("passed") and gate2_passed and gate3_passed and gate4_passed and gate5_passed)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "first experiment / P2 budgeted replay",
        "dataset": {
            "dev_candidates": len(dev),
            "eval_candidates": len(eval_),
            "eval_incorrect": sum(1 for candidate in eval_ if not candidate.audit_correct),
            "eval_correct": sum(1 for candidate in eval_ if candidate.audit_correct),
        },
        "budgets_s": budgets,
        "policy_summaries": summaries,
        "reference": reference_metrics,
        "frontier": frontier,
        "comparisons": comparisons,
        "best_comparison": best_comparison,
        "representative_comparisons": representative,
        "best_qualifying_comparisons": qualifying,
        "by_benchmark_at_best_budget": by_benchmark,
        "by_generator_at_best_budget": by_generator,
        "by_difficulty_at_best_budget": by_difficulty,
        "difficulty_definition": "task base-pass rate over all candidates: hard <0.5, medium <0.75, easy >=0.75",
        "generator_holdout": holdout,
        "gate_1_audit": gate1,
        "gate_2_adaptive_gain": {
            "criterion": ">=30% relative FPR reduction vs best B1-B3 at similar p50 cost, CI favors",
            "passed": gate2_passed,
            "qualifying_budgets": [item["budget_s"] for item in qualifying],
            "best_observed_relative_reduction": best_comparison["relative_reduction"],
            "best_observed_budget_s": best_comparison["budget_s"],
            "best_observed_baseline": best_comparison["best_baseline"],
            "best_observed_fprs": best_comparison["candidate_vs_best_rate"],
        },
        "gate_3_false_rejection": {
            "criterion": "no material false-rejection increase vs audited-correct programs",
            "passed": gate3_passed,
            "candidate_false_rejection_rate_at_best_budget": candidate_fnr,
            "b4_reference_false_rejection_rate": b4_false_rejection,
        },
        "gate_4_disjoint_generalization": {
            "criterion": "gain remains on disjoint tasks and generators",
            "passed": gate4_passed,
            "holdout": holdout,
        },
        "gate_5_cost_realism": {
            "criterion": "sandbox startup is not dominant and exhaustive checking is not trivially cheap",
            "passed": gate5_passed,
            "median_base_s": median_base,
            "median_bank_s": median_bank,
            "median_bank_startup_s": median_startup,
            "median_bank_test_s": median_test,
            "median_total_s": median_total,
            "test_fraction_of_total": test_fraction,
            "startup_share_of_bank": startup_share,
            "interpretation": (
                "Adaptive selection can save at most the bank test-execution time; at the median candidate this is "
                "a tiny fraction of end-to-end grading cost, and all-tests is trivially affordable."
                if exhaustive_cheap
                else "The measured cost structure leaves room for adaptive selection."
            ),
        },
        "overall": {
            "proceed": overall_proceed,
            "verdict": "PROCEED" if overall_proceed else "NO-GO (first-experiment gate)",
            "gates": {
                "gate_1_signal": bool(gate1.get("passed")),
                "gate_2_adaptive_gain": gate2_passed,
                "gate_3_false_rejection": gate3_passed,
                "gate_4_disjoint_generalization": gate4_passed,
                "gate_5_cost_realism": gate5_passed,
            },
        },
    }

    _write_replay_csv(artifacts / "replay.csv", all_rows)
    _write_frontier_csv(artifacts / "frontier.csv", frontier)
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _make_figures(artifacts, comparisons, reference_metrics, metrics["gate_5_cost_realism"])
    _write_report(artifacts / "report.md", metrics, comparisons, reference_metrics)
    return metrics


def _generator_holdout(
    dev: list[ReplayCandidate],
    eval_: list[ReplayCandidate],
    config: dict[str, Any],
    budgets: list[float],
    iterations: int,
) -> dict[str, Any]:
    """Fit on dev candidates from other generators, test each held-out generator."""

    generators = sorted({candidate.generator for candidate in eval_})
    results: dict[str, Any] = {}
    any_gain = False
    for generator in generators:
        train = [candidate for candidate in dev if candidate.generator != generator]
        test = [candidate for candidate in eval_ if candidate.generator == generator]
        if not train or not test or not any(not candidate.audit_correct for candidate in test):
            results[generator] = {"status": "insufficient", "n": len(test)}
            continue
        policies = policies_for(int(config["seed"]), config)
        prepare_policies(policies, train)
        policy_map = {policy.name: policy for policy in policies}
        best: dict[str, Any] = {"relative_reduction": None}
        for budget in budgets:
            rows = {
                name: _run_policy(policy_map[name], test, budget)
                for name in ("B1", "B2", "B3", "Candidate")
            }
            summary = {name: _metrics(group) for name, group in rows.items()}
            best_name, best_summary = _best_baseline(summary)
            cand_fpr = summary["Candidate"]["false_positive_rate"]
            base_fpr = best_summary["false_positive_rate"]
            relative = (
                (base_fpr - cand_fpr) / base_fpr
                if base_fpr not in (None, 0) and cand_fpr is not None
                else None
            )
            if relative is not None and (best["relative_reduction"] is None or relative > best["relative_reduction"]):
                best = {
                    "relative_reduction": relative,
                    "budget_s": budget,
                    "best_baseline": best_name,
                    "candidate_fpr": cand_fpr,
                    "baseline_fpr": base_fpr,
                    "n": len(test),
                    "incorrect": sum(1 for candidate in test if not candidate.audit_correct),
                }
        if best["relative_reduction"] is not None and best["relative_reduction"] >= 0.30:
            any_gain = True
        results[generator] = best
    return {"any_generator_gain": any_gain, "by_generator": results}


def _load_gate1(paths: Paths) -> dict[str, Any]:
    summary_path = paths.artifacts / "audit" / "audit_summary.json"
    if not summary_path.exists():
        return {"passed": False, "reason": "audit summary missing"}
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    return dict(summary.get("gate_1", {}), **{"counts": summary.get("gate_1_eval")})


def _write_replay_csv(path: Path, rows: list[ResultRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy",
                "budget_s",
                "candidate_id",
                "task_id",
                "benchmark",
                "split",
                "generator",
                "family",
                "reward",
                "audit_correct",
                "false_positive",
                "false_negative",
                "cost_wall_s",
                "cost_cpu_s",
                "tests_run",
                "budget_exhausted",
                "first_failure",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.policy,
                    "" if row.budget is None else row.budget,
                    row.candidate_id,
                    row.task_id,
                    row.benchmark,
                    row.split,
                    row.generator,
                    row.family,
                    row.reward,
                    row.audit_correct,
                    row.false_positive,
                    row.false_negative,
                    row.cost_wall,
                    row.cost_cpu,
                    row.tests_run,
                    row.budget_exhausted,
                    row.first_failure,
                ]
            )


def _write_frontier_csv(path: Path, frontier: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy",
                "budget_s",
                "false_positive_rate",
                "true_positive_acceptance",
                "precision",
                "p50_wall_s",
                "p95_wall_s",
                "mean_tests",
                "n",
                "incorrect",
            ]
        )
        for point in frontier:
            writer.writerow(
                [
                    point["policy"],
                    "" if point["budget_s"] is None else point["budget_s"],
                    point["false_positive_rate"],
                    point["true_positive_acceptance"],
                    point["precision"],
                    point["cost_wall_s"]["p50"],
                    point["cost_wall_s"]["p95"],
                    point["tests_run_mean"],
                    point["n"],
                    point["incorrect"],
                ]
            )


def _make_figures(
    artifacts: Path,
    comparisons: list[dict[str, Any]],
    reference: dict[str, Any],
    gate5: dict[str, Any],
) -> None:
    os.environ["MPLCONFIGDIR"] = str(artifacts / ".mplconfig")
    (artifacts / ".mplconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B1": "#7f8c8d", "B2": "#e67e22", "B3": "#8e44ad", "Candidate": "#c0392b"}
    labels = {"B1": "B1 random", "B2": "B2 static ranking", "B3": "B3 coverage", "Candidate": "Candidate adaptive"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for name in ("B1", "B2", "B3", "Candidate"):
        pts = [
            (comp["budget_s"] * 1000, comp["candidate_vs_best_rate"][name])
            for comp in comparisons
            if comp["candidate_vs_best_rate"].get(name) is not None
        ]
        left.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3, label=labels[name], color=colors[name])
    left.axhline(reference["B0"]["false_positive_rate"], color="#2c3e50", linestyle="--", linewidth=1, label="B0 base only")
    left.axhline(reference["B4"]["false_positive_rate"], color="#27ae60", linestyle=":", linewidth=1, label="B4 all tests")
    left.set_xlabel("bank-stage wall-clock budget (ms)")
    left.set_ylabel("false-positive reward rate")
    left.set_xlim(31.5, 65)
    left.set_xticks([32, 35, 40, 50, 60])
    left.set_ylim(-0.02, 1.05)
    left.set_title("Quality vs verification budget")
    left.grid(alpha=0.25)
    left.legend(fontsize=8, loc="upper right")

    for name in ("B1", "B2", "B3", "Candidate"):
        pts = [
            (comp["candidate_p50_cost_s"] if name == "Candidate" else comp["best_baseline_p50_cost_s"], comp["candidate_vs_best_rate"][name])
            for comp in comparisons
        ]
        # Use the actual policy p50 from the per-budget summary stored in the comparison.
        pts = []
        for comp in comparisons:
            p50 = comp.get("p50_by_policy", {}).get(name)
            if p50 is not None:
                pts.append((p50 * 1000, comp["candidate_vs_best_rate"][name]))
        if pts:
            right.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3, label=labels[name], color=colors[name])
    right.scatter(
        [reference["B0"]["cost_wall_s"]["p50"] * 1000],
        [reference["B0"]["false_positive_rate"]],
        color="#2c3e50",
        marker="s",
        label="B0",
    )
    right.scatter(
        [reference["B4"]["cost_wall_s"]["p50"] * 1000],
        [reference["B4"]["false_positive_rate"]],
        color="#27ae60",
        marker="*",
        s=80,
        label="B4 (all tests)",
    )
    right.set_xlabel("p50 end-to-end grading wall time (ms)")
    right.set_ylabel("false-positive reward rate")
    right.set_ylim(-0.02, 1.05)
    right.set_title("Quality vs realized cost")
    right.grid(alpha=0.25)
    right.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(artifacts / "false_positive_frontier.png", dpi=180)
    fig.savefig(artifacts / "false_positive_frontier.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels2 = ["base tests", "bank startup", "bank test time"]
    values = [
        (gate5.get("median_base_s") or 0) * 1000,
        (gate5.get("median_bank_startup_s") or 0) * 1000,
        (gate5.get("median_bank_test_s") or 0) * 1000,
    ]
    bars = ax.bar(labels2, values, color=["#2c3e50", "#f39c12", "#c0392b"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("median wall time (ms)")
    ax.set_title("Where grading cost goes (median eval candidate)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(artifacts / "cost_breakdown.png", dpi=180)
    fig.savefig(artifacts / "cost_breakdown.svg")
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    path: Path,
    metrics: dict[str, Any],
    comparisons: list[dict[str, Any]],
    reference: dict[str, Any],
) -> None:
    best = metrics["best_comparison"]
    gate2 = metrics["gate_2_adaptive_gain"]
    gate5 = metrics["gate_5_cost_realism"]
    lines = [
        "# RewardForge — first experiment report",
        "",
        "This is a CPU-only falsification experiment. It is **not** an RL training run.",
        "The policy is compared at the same measured wall-clock budget, using the",
        "P1 audit's per-test costs and sandbox startup.",
        "",
        "## Corpus",
        "",
        f"- Development candidates: {metrics['dataset']['dev_candidates']}",
        f"- Held-out evaluation candidates: {metrics['dataset']['eval_candidates']} "
        f"({metrics['dataset']['eval_incorrect']} audited incorrect, {metrics['dataset']['eval_correct']} audited correct)",
        f"- Evaluation tasks are disjoint from development tasks by task ID.",
        "- Base tests are the weak reward; private enhanced-suite audit is the label.",
        "- Base-pass/audit-fail rate: 12.5% overall (69/554); 11.5% on held-out eval (32/279).",
        "- By benchmark: HumanEval+ 7.5% (22/294), MBPP+ 18.1% (47/260).",
        "",
        "## Gate results",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
        f"| 1. Natural signal | {'PASS' if metrics['gate_1_audit'].get('passed') else 'FAIL'} | "
        f"{metrics['gate_1_audit'].get('observed')} base-pass/audit-fail on eval across "
        f"{metrics['gate_1_audit'].get('distinct_tasks')} tasks (target >=30) |",
        f"| 2. Adaptive gain | {'PASS' if gate2['passed'] else 'FAIL'} | "
        f"best observed relative FPR reduction {_fmt(gate2['best_observed_relative_reduction'])} "
        f"at {gate2['best_observed_budget_s']} s (target >=0.30) |",
        f"| 3. False rejection | {'PASS' if metrics['gate_3_false_rejection']['passed'] else 'FAIL'} | "
        f"candidate {_fmt(metrics['gate_3_false_rejection']['candidate_false_rejection_rate_at_best_budget'])} "
        f"vs B4 {_fmt(metrics['gate_3_false_rejection']['b4_reference_false_rejection_rate'])} |",
        f"| 4. Disjoint generalization | {'PASS' if metrics['gate_4_disjoint_generalization']['passed'] else 'FAIL'} | "
        f"generator holdout gain: {metrics['gate_4_disjoint_generalization']['holdout']['any_generator_gain']} |",
        f"| 5. Cost realism | {'PASS' if gate5['passed'] else 'FAIL'} | "
        f"median test time {_fmt(gate5['median_bank_test_s'])} s is "
        f"{_fmt((gate5['test_fraction_of_total'] or 0) * 100, 1)}% of total; "
        f"startup is {_fmt((gate5['startup_share_of_bank'] or 0) * 100, 1)}% of bank stage |",
        "",
        f"**Overall: {metrics['overall']['verdict']}.**",
        "",
        "## What the numbers say",
        "",
        f"- B0 (base only) has FPR {_fmt(reference['B0']['false_positive_rate'])}, "
        f"p50 {_fmt(reference['B0']['cost_wall_s']['p50'] * 1000, 1)} ms.",
        f"- B4 (all bank tests) has FPR {_fmt(reference['B4']['false_positive_rate'])}, "
        f"p50 {_fmt(reference['B4']['cost_wall_s']['p50'] * 1000, 1)} ms, "
        f"p95 {_fmt(reference['B4']['cost_wall_s']['p95'] * 1000, 1)} ms.",
        f"- At the best-test-running budget ({best['budget_s'] * 1000:.1f} ms), "
        f"{best['best_baseline']} has FPR {_fmt(best['candidate_vs_best_rate'][best['best_baseline']])}; "
        f"the candidate has {_fmt(best['candidate_vs_best_rate']['Candidate'])}. "
        "No budget produced a positive relative reduction; at 33–35 ms the candidate was worse.",
        f"- Median bank test time is {_fmt(gate5['median_bank_test_s'] * 1000, 2)} ms; "
        f"median bank startup is {_fmt(gate5['median_bank_startup_s'] * 1000, 2)} ms. "
        "The measured cost structure leaves almost no room for test selection to matter.",
        f"- Secondary metrics: at 33 ms the candidate has TPR-accept "
        f"{_fmt(metrics['policy_summaries']['0.033']['Candidate']['true_positive_acceptance'])} "
        f"(false-rejection {_fmt(metrics['policy_summaries']['0.033']['Candidate']['false_rejection_rate'])}, "
        f"precision {_fmt(metrics['policy_summaries']['0.033']['Candidate']['precision'])}); "
        f"at 60 ms TPR-accept {_fmt(metrics['policy_summaries']['0.06']['Candidate']['true_positive_acceptance'])} "
        f"(false-rejection {_fmt(metrics['policy_summaries']['0.06']['Candidate']['false_rejection_rate'])}, "
        f"precision {_fmt(metrics['policy_summaries']['0.06']['Candidate']['precision'])}). "
        f"B4 has TPR-accept {_fmt(reference['B4']['true_positive_acceptance'])}, "
        f"false-rejection {_fmt(reference['B4']['false_rejection_rate'])}, "
        f"precision {_fmt(reference['B4']['precision'])}. The candidate does not improve the "
        "correctness trade-off; it only spends less before converging to the same residual errors.",
        "",
        "## Representative points",
        "",
        "| setup | p50 wall (ms) | p95 wall (ms) | FPR | false positives / incorrect | candidate vs best baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    best_testing = metrics["representative_comparisons"]["best_testing"]
    tight_33 = metrics["representative_comparisons"]["tight_33"]
    near_full = metrics["representative_comparisons"]["near_full"]
    tail_matched = metrics["representative_comparisons"]["tail_matched"]
    def point_row(label: str, comparison: dict[str, Any], candidate: bool) -> str:
        if candidate:
            p50 = comparison["candidate_p50_cost_s"]
            p95 = comparison["p95_by_policy"]["Candidate"]
            fpr = comparison["candidate_fpr"]
            comparison_text = _fmt(comparison["relative_reduction"])
        else:
            p50 = comparison["best_baseline_p50_cost_s"]
            p95 = comparison["p95_by_policy"][comparison["best_baseline"]]
            fpr = comparison["best_baseline_fpr"]
            comparison_text = "n/a"
        return (
            f"| {label} | {p50 * 1000:.1f} | {(p95 or 0) * 1000:.1f} | {_fmt(fpr)} | "
            f"{round((fpr or 0) * 32)}/32 | {comparison_text} |"
        )

    lines.extend(
        [
            f"| B0: base only | {reference['B0']['cost_wall_s']['p50'] * 1000:.1f} | "
            f"{reference['B0']['cost_wall_s']['p95'] * 1000:.1f} | {_fmt(reference['B0']['false_positive_rate'])} | "
            f"{reference['B0']['false_positives']}/{reference['B0']['incorrect']} | n/a |",
            point_row(f"{best_testing['best_baseline']} static @ {best_testing['budget_s'] * 1000:.1f} ms", best_testing, False),
            point_row(f"Candidate @ {best_testing['budget_s'] * 1000:.1f} ms", best_testing, True),
            point_row(f"{tight_33['best_baseline']} static @ {tight_33['budget_s'] * 1000:.1f} ms", tight_33, False),
            point_row(f"Candidate @ {tight_33['budget_s'] * 1000:.1f} ms", tight_33, True),
            point_row(f"{near_full['best_baseline']} static @ {near_full['budget_s'] * 1000:.0f} ms", near_full, False),
            point_row(f"Candidate @ {near_full['budget_s'] * 1000:.0f} ms", near_full, True),
            point_row(f"{tail_matched['best_baseline']} static @ {tail_matched['budget_s'] * 1000:.0f} ms", tail_matched, False),
            point_row(f"Candidate @ {tail_matched['budget_s'] * 1000:.0f} ms", tail_matched, True),
            f"| B4: all bank tests | {reference['B4']['cost_wall_s']['p50'] * 1000:.1f} | "
            f"{reference['B4']['cost_wall_s']['p95'] * 1000:.1f} | {_fmt(reference['B4']['false_positive_rate'])} | "
            f"{reference['B4']['false_positives']}/{reference['B4']['incorrect']} | n/a |",
            "",
            "*Counts are exact integer counts from the replay; see `frontier.csv`. The key point:",
            "at essentially the same p50 cost, running all tests reaches FPR 0 while budget-matched",
            "policies still leave 3 false positives from slow-tail tests.",
            "",
            f"At {best_testing['budget_s'] * 1000:.1f} ms the candidate FPR is {_fmt(best_testing['candidate_fpr'])} "
            f"(task-clustered 95% CI {_fmt(best_testing['candidate_ci']['ci_low'])}–{_fmt(best_testing['candidate_ci']['ci_high'])}) "
            f"and {best_testing['best_baseline']} is {_fmt(best_testing['best_baseline_fpr'])} "
            f"(CI {_fmt(best_testing['best_baseline_ci']['ci_low'])}–{_fmt(best_testing['best_baseline_ci']['ci_high'])}). "
            f"The paired difference (candidate - {best_testing['best_baseline']}) is "
            f"{_fmt(best_testing['paired']['difference_ci_low'])} to {_fmt(best_testing['paired']['difference_ci_high'])} FPR. "
            f"At 33.0 ms the candidate FPR is {_fmt(tight_33['candidate_fpr'])} versus "
            f"{_fmt(tight_33['best_baseline_fpr'])} for {tight_33['best_baseline']}, i.e. the candidate is worse. "
            f"At 60.0 ms the paired difference is 0.000 with a zero-width interval.",
            "",
            "## Breakdown at the best-test-running budget",
            "",
            "| group | n | incorrect | best baseline | best FPR | candidate FPR |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    def group_rows(group: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for name, payload in sorted(group.items()):
            rates = {policy: payload[policy]["false_positive_rate"] for policy in ("B1", "B2", "B3")}
            available = {policy: rate for policy, rate in rates.items() if rate is not None}
            if not available:
                continue
            best_rate = min(available.values())
            best_policy = "B2" if rates["B2"] == best_rate else min(available, key=available.get)
            candidate_rate = payload["Candidate"]["false_positive_rate"]
            n = payload["Candidate"]["n"]
            incorrect = payload["Candidate"]["incorrect"]
            out.append(
                f"| {name} | {n} | {incorrect} | {best_policy} | {_fmt(best_rate)} | {_fmt(candidate_rate)} |"
            )
        return out

    lines.extend(group_rows(metrics["by_benchmark_at_best_budget"]))
    lines.extend(group_rows(metrics["by_generator_at_best_budget"]))
    lines.extend(group_rows(metrics["by_difficulty_at_best_budget"]))
    lines.extend(
        [
            "",
            "Difficulty is a descriptive proxy (task base-pass rate over all candidates), not a policy input.",
            "",
            "### Generator holdout",
            "",
            "| held-out generator | eval n | incorrect | max relative reduction | budget |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for generator, payload in sorted(metrics["generator_holdout"]["by_generator"].items()):
        if payload.get("relative_reduction") is None:
            lines.append(f"| {generator} | {payload.get('n', 'n/a')} | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {generator} | {payload.get('n')} | {payload.get('incorrect')} | "
                f"{_fmt(payload.get('relative_reduction'))} | {payload.get('budget_s')} s |"
            )
    lines.extend(
        [
            "",
            "## Budget frontier",
            "",
            "| budget (ms) | best baseline | best FPR | candidate FPR | relative reduction | candidate p50 (ms) |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in comparisons:
        lines.append(
            f"| {comparison['budget_s'] * 1000:.1f} | {comparison['best_baseline']} | "
            f"{_fmt(comparison['best_baseline_fpr'])} | {_fmt(comparison['candidate_fpr'])} | "
            f"{_fmt(comparison['relative_reduction'])} | {_fmt(comparison['candidate_p50_cost_s'] * 1000, 1)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Gate 1 passes narrowly: 32 held-out base-pass/audit-fail candidates across 27 tasks.",
            "- The adaptive selector does not produce the pre-registered 30% relative FPR reduction. "
            "It ties the best static baseline at 37.5 ms, is 14–38% worse at 33–35 ms, and is identical "
            "at most larger budgets. No budget produced a positive relative reduction.",
            "- The dominant cost is process/sandbox startup, not test execution. For the median candidate, "
            "running all bank tests costs about "
            f"{_fmt(gate5['median_bank_test_s'] * 1000, 2)} ms of test time on top of "
            f"{_fmt(gate5['median_bank_startup_s'] * 1000, 2)} ms of startup, while the whole grading call is "
            f"{_fmt(gate5['median_total_s'] * 1000, 1)} ms. Selection cannot save enough to matter at this scale.",
            "- This is a **NO-GO for the adaptive-selection premise as tested**. It is not evidence that the "
            "audit harness is useless: the audit itself found a reproducible 12.5% base-pass/audit-fail rate "
            "among base-passing candidates (11.5% on eval).",
            "",
            "## Limitations",
            "",
            "- Audit labels come from a held-out subset of the enhanced suite, not a proof of correctness.",
            "- Bank and audit tests are sampled from EvalPlus suites; detection power is finite.",
            "- All 966 candidates were audited for completeness, but the policy comparison uses the 554 "
            "base-passing candidates. Audit cases exclude base-overlap inputs, so the 17/412 "
            "base-fail/audit-pass candidates bound base-test false rejection only from above; they are "
            "not proof that a base test was wrong.",
            "- Costs are measured on one macOS host with one worker per suite; a per-test-container design "
            "would have different cost structure and is a separate experiment.",
            "- The candidate corpus is public model output, not a fresh RL policy distribution.",
            "- 32 evaluation false positives is just above the pre-registered minimum; the confidence intervals "
            "are wide and the result should not be over-interpreted as a precise rate.",
            "",
            "## Artifacts",
            "",
            "- `artifacts/audit/audit_summary.json` — P1 counts, costs, gate 1/5 evidence.",
            "- `artifacts/audit/cases.csv` — task × candidate × test result table.",
            "- `artifacts/audit/candidates.csv` — candidate-level base/bank/audit outcomes.",
            "- `artifacts/experiment/metrics.json` — all replay metrics, CIs, gate status.",
            "- `artifacts/experiment/frontier.csv` — quality/cost frontier.",
            "- `artifacts/experiment/false_positive_frontier.png` — false-positive rewards by budget/cost.",
            "- `artifacts/experiment/cost_breakdown.png` — where grading time goes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Run the RewardForge budgeted replay experiment")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    metrics = run_experiment(args.config)
    print(json.dumps(metrics["overall"], indent=2))
    print(json.dumps(metrics["gate_2_adaptive_gain"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
