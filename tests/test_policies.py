from __future__ import annotations

from rewardforge.policies import (
    AdaptivePolicy,
    B2Policy,
    B3Policy,
    ReplayCandidate,
    ReplayCase,
    replay_candidate,
)


def _case(index: int, cluster: str, status: str = "pass", cost: float = 0.001) -> ReplayCase:
    return ReplayCase(
        index=index,
        source_index=index,
        cluster=cluster,
        coverage=(f"cluster:{cluster}",),
        cost=cost,
        cpu=cost,
        status=status,
        failure_kind="wrong_output" if status != "pass" else "",
    )


def _candidate(candidate_id: str, cases: list[ReplayCase], correct: bool) -> ReplayCandidate:
    return ReplayCandidate(
        candidate_id=candidate_id,
        task_id="T/1",
        benchmark="humaneval",
        split="dev",
        generator="g",
        family="g",
        base_wall=0.05,
        base_cpu=0.01,
        bank_startup=0.03,
        bank_startup_cpu=0.005,
        cases=cases,
        audit_correct=correct,
        bank_all_passed=all(case.status == "pass" for case in cases),
    )


def test_b2_prefers_high_failure_cluster():
    dev = [
        _candidate("d1", [_case(0, "hard", "fail"), _case(1, "easy")], False),
        _candidate("d2", [_case(0, "hard", "fail"), _case(1, "easy")], False),
        _candidate("d3", [_case(0, "hard"), _case(1, "easy")], True),
    ]
    policy = B2Policy()
    policy.prepare(dev)
    state = policy.new_state(_candidate("e1", [_case(0, "easy"), _case(1, "hard")], True))
    assert policy.choose(_candidate("e1", state["order"], True), state, state["order"]).cluster == "hard"


def test_b3_coverage_and_adaptive_do_not_crash():
    dev = [
        _candidate("d1", [_case(0, "hard", "fail"), _case(1, "easy")], False),
        _candidate("d2", [_case(0, "hard", "fail"), _case(2, "other", "fail")], False),
        _candidate("d3", [_case(0, "hard"), _case(1, "easy")], True),
    ]
    for policy in (B3Policy(), AdaptivePolicy()):
        policy.prepare(dev)
        candidate = _candidate("e1", [_case(0, "easy"), _case(1, "hard")], False)
        state = policy.new_state(candidate)
        assert policy.choose(candidate, state, state.get("order", candidate.cases)) is not None


def test_replay_stops_on_failure_and_respects_budget():
    candidate = _candidate(
        "e1",
        [_case(0, "a", "fail"), _case(1, "b", "pass")],
        False,
    )
    policy = B2Policy()
    policy.prepare([_candidate("d1", [_case(0, "a", "fail")], False)])
    result = replay_candidate(policy, candidate, 1.0)
    assert result.reward == 0
    assert result.tests_run == 1

    tiny = replay_candidate(policy, candidate, 0.0)
    assert tiny.reward == 1
    assert tiny.budget_exhausted
    assert tiny.tests_run == 0


def test_b4_runs_all_available_tests():
    candidate = _candidate("e1", [_case(0, "a"), _case(1, "b")], True)
    from rewardforge.policies import B4Policy

    result = replay_candidate(B4Policy(), candidate, None)
    assert result.reward == 1
    assert result.tests_run == 2
