from __future__ import annotations

from rewardforge.experiment import ResultRow, _best_baseline, _metrics


def _row(policy: str, correct: bool, reward: int, cost: float = 0.1) -> ResultRow:
    return ResultRow(
        policy=policy,
        budget=0.01,
        candidate_id=f"{policy}:{correct}:{reward}",
        task_id="T/1",
        benchmark="humaneval",
        split="eval",
        generator="g",
        family="g",
        reward=reward,
        audit_correct=correct,
        cost_wall=cost,
        cost_cpu=cost / 2,
        tests_run=3,
        budget_exhausted=False,
        first_failure="",
    )


def test_false_positive_metrics():
    rows = [
        _row("B2", False, 1),   # false positive
        _row("B2", False, 0),   # true negative
        _row("B2", True, 1),    # true positive
        _row("B2", True, 0),    # false negative
    ]
    metrics = _metrics(rows)
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["true_positive_acceptance"] == 0.5
    assert metrics["false_rejection_rate"] == 0.5
    assert metrics["precision"] == 0.5


def test_best_baseline_tie_prefers_b2():
    summary = {
        "B1": {"false_positive_rate": 0.2},
        "B2": {"false_positive_rate": 0.1},
        "B3": {"false_positive_rate": 0.1},
    }
    name, _ = _best_baseline(summary)
    assert name == "B2"
