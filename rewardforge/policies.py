"""Budgeted policies and offline replay.

The policy only sees legal pre-grade observations: dev-split bank outcomes,
test features, measured costs, and the current candidate's base/bank
observations.  It never sees private audit labels or audit cases.

B0  base tests only
B1  random valid bank order, stop on failure
B2  strongest static ranking from dev (cluster failure rate / cost)
B3  dev-only greedy coverage / set-cover ranking
C   sequential adaptive mixture over dev failure profiles
B4  all bank tests (cost/quality reference, not budget-matched)
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import Paths, stable_seed
from .dataio import load_tasks


@dataclass
class ReplayCase:
    index: int
    source_index: int
    cluster: str
    coverage: tuple[str, ...]
    cost: float
    cpu: float
    status: str  # pass | fail | timeout
    failure_kind: str


@dataclass
class ReplayCandidate:
    candidate_id: str
    task_id: str
    benchmark: str
    split: str
    generator: str
    family: str
    base_wall: float
    base_cpu: float
    bank_startup: float
    bank_startup_cpu: float
    cases: list[ReplayCase]
    audit_correct: bool
    bank_all_passed: bool


def _coverage_keys(features: dict[str, Any]) -> tuple[str, ...]:
    keys = {
        f"cluster:{features.get('cluster', '')}",
        f"size:{features.get('size_bucket', '')}",
        f"arity:{features.get('arity', '')}",
        f"ret:{features.get('expected_type', '')}",
    }
    for flag in features.get("edge_flags", []):
        keys.add(f"edge:{flag}")
    return tuple(sorted(keys))


def build_replay_data(paths: Paths | None = None) -> tuple[list[ReplayCandidate], list[ReplayCandidate]]:
    paths = paths or Paths()
    tasks = load_tasks(paths)
    records: list[dict[str, Any]] = []
    with open(paths.artifacts / "audit" / "matrix.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    candidates: list[ReplayCandidate] = []
    for record in records:
        if not record["eligible_for_positive_reward"]:
            continue
        if record["excluded"] or record.get("audited_correct") is None:
            continue
        task = tasks[record["task_id"]]
        bank_outcomes = {outcome["index"]: outcome for outcome in record["bank"]["outcomes"]}
        cases: list[ReplayCase] = []
        for index, case in enumerate(task["bank_cases"]):
            outcome = bank_outcomes.get(index)
            if outcome is None or outcome["status"] == "not_run":
                # The worker stopped before this test (earlier timeout/crash).
                # The true outcome is unobserved, so it is not selectable in
                # replay.  At least one timeout/failure remains selectable.
                continue
            status = outcome["status"]
            if status == "harness_error":
                continue
            if status not in {"pass", "fail", "timeout", "crash"}:
                continue
            cases.append(
                ReplayCase(
                    index=index,
                    source_index=int(outcome.get("source_index", case.get("source_index", index))),
                    cluster=str(case["features"].get("cluster", "")),
                    coverage=_coverage_keys(case["features"]),
                    cost=float(outcome["wall_s"]),
                    cpu=float(outcome["cpu_s"]),
                    status="pass" if status == "pass" else "fail",
                    failure_kind=outcome["failure_kind"],
                )
            )
        if not cases:
            continue
        candidates.append(
            ReplayCandidate(
                candidate_id=record["candidate_id"],
                task_id=record["task_id"],
                benchmark=record["benchmark"],
                split=record["split"],
                generator=record["generator"],
                family=record["family"],
                base_wall=float(record["base"]["wall_s"]),
                base_cpu=float(record["base"]["cpu_s"]),
                bank_startup=float(record["bank"]["startup_s"]),
                bank_startup_cpu=float(record["bank"]["startup_cpu_s"]),
                cases=cases,
                audit_correct=bool(record["audited_correct"]),
                bank_all_passed=bool(record["bank"]["all_passed"]),
            )
        )
    dev = [candidate for candidate in candidates if candidate.split == "dev"]
    eval_ = [candidate for candidate in candidates if candidate.split == "eval"]
    return dev, eval_


# ---------------------------------------------------------------------------
# Development-only statistics
# ---------------------------------------------------------------------------


@dataclass
class ClusterStat:
    runs: int = 0
    failures: int = 0
    cost_sum: float = 0.0

    @property
    def failure_rate(self) -> float:
        return (self.failures + 1.0) / (self.runs + 2.0)

    @property
    def mean_cost(self) -> float:
        return self.cost_sum / self.runs if self.runs else 1e-6


def cluster_stats(candidates: Iterable[ReplayCandidate]) -> dict[str, ClusterStat]:
    stats: dict[str, ClusterStat] = defaultdict(ClusterStat)
    for candidate in candidates:
        for case in candidate.cases:
            stat = stats[case.cluster]
            stat.runs += 1
            stat.failures += 1 if case.status != "pass" else 0
            stat.cost_sum += case.cost
    return dict(stats)


def coverage_weights(candidates: Iterable[ReplayCandidate]) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        for case in candidate.cases:
            if case.status != "pass":
                for key in case.coverage:
                    weights[key] += 1.0
    return dict(weights)


@dataclass
class FailureProfile:
    clusters: set[str]
    failure_rates: dict[str, float]
    weight: float


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def build_failure_profiles(
    dev: list[ReplayCandidate], k: int = 8
) -> tuple[list[FailureProfile], float]:
    failing: list[list[ReplayCase]] = []
    for candidate in dev:
        failures = [case for case in candidate.cases if case.status != "pass"]
        if failures:
            failing.append(failures)
    if not failing:
        return [], 0.0
    sets = [set(case.cluster for case in failures) for failures in failing]
    # Sort the unique failure signatures so the greedy medoids, profile order,
    # and adaptive scoring are independent of Python's hash randomization.
    unique = sorted({frozenset(item) for item in sets}, key=lambda item: tuple(sorted(item)))
    if len(unique) <= k:
        medoids = [set(item) for item in unique]
    else:
        # Deterministic greedy k-medoids on Jaccard distance.
        medoids = [set(max(unique, key=len))]
        while len(medoids) < k:
            best_index = None
            best_distance = -1.0
            for index, item in enumerate(unique):
                distance = min(1.0 - _jaccard(set(item), medoid) for medoid in medoids)
                if distance > best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                break
            medoids.append(set(unique[best_index]))

    assignments: list[list[list[ReplayCase]]] = [[] for _ in medoids]
    for failures in failing:
        case_set = set(case.cluster for case in failures)
        distances = [1.0 - _jaccard(case_set, medoid) for medoid in medoids]
        assignments[min(range(len(medoids)), key=lambda index: distances[index])].append(failures)

    profiles: list[FailureProfile] = []
    total = sum(len(group) for group in assignments) or 1
    for medoid, group in zip(medoids, assignments):
        if not group:
            continue
        # Compute rates from the raw dev candidate cases belonging to this
        # profile's members.  Membership is by profile medoid, approximated by
        # the candidate's failed-cluster set.
        member_sets = {frozenset(set(case.cluster for case in failures)) for failures in group}
        runs: Counter[str] = Counter()
        fails: Counter[str] = Counter()
        for candidate in dev:
            candidate_set = frozenset(case.cluster for case in candidate.cases if case.status != "pass")
            if candidate_set not in member_sets:
                continue
            for case in candidate.cases:
                runs[case.cluster] += 1
                if case.status != "pass":
                    fails[case.cluster] += 1
        profiles.append(
            FailureProfile(
                clusters=set(medoid),
                failure_rates={cluster: (fails[cluster] + 1.0) / (runs[cluster] + 2.0) for cluster in set(runs) | set(fails)},
                weight=len(group) / total,
            )
        )
    p_buggy = len(failing) / len(dev) if dev else 0.0
    return profiles, p_buggy


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class Policy:
    name = "policy"

    def prepare(self, dev: list[ReplayCandidate]) -> None:
        pass

    def new_state(self, candidate: ReplayCandidate) -> Any:
        return None

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        raise NotImplementedError

    def observe(self, candidate: ReplayCandidate, state: Any, case: ReplayCase, passed: bool) -> None:
        pass


class B0Policy(Policy):
    name = "B0"

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        raise RuntimeError("B0 has no bank tests")


class B1Policy(Policy):
    name = "B1"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def new_state(self, candidate: ReplayCandidate) -> Any:
        order = list(candidate.cases)
        random.Random(stable_seed(self.seed, candidate.candidate_id, "B1")).shuffle(order)
        return {"order": order}

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        available = {case.index for case in remaining}
        for case in state["order"]:
            if case.index in available:
                return case
        return remaining[0]


class B2Policy(Policy):
    name = "B2"

    def __init__(self) -> None:
        self.stats: dict[str, ClusterStat] = {}
        self.global_order: dict[str, int] = {}

    def prepare(self, dev: list[ReplayCandidate]) -> None:
        self.stats = cluster_stats(dev)
        ranked = sorted(
            self.stats.items(),
            key=lambda item: (-item[1].failure_rate / max(item[1].mean_cost, 1e-9), item[0]),
        )
        self.global_order = {cluster: index for index, (cluster, _) in enumerate(ranked)}

    def new_state(self, candidate: ReplayCandidate) -> Any:
        order = sorted(
            candidate.cases,
            key=lambda case: (
                self.global_order.get(case.cluster, 10**9),
                -self.stats.get(case.cluster, ClusterStat()).failure_rate,
                case.cost,
                case.index,
            ),
        )
        return {"order": order}

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        available = {case.index for case in remaining}
        for case in state["order"]:
            if case.index in available:
                return case
        return remaining[0]


class B3Policy(Policy):
    name = "B3"

    def __init__(self) -> None:
        self.cluster: dict[str, ClusterStat] = {}
        self.coverage: dict[str, float] = {}
        self.ranked_clusters: list[str] = []
        self.global_order: dict[str, int] = {}

    def prepare(self, dev: list[ReplayCandidate]) -> None:
        self.cluster = cluster_stats(dev)
        self.coverage = coverage_weights(dev)
        self.ranked_clusters = [
            cluster
            for cluster, _ in sorted(
                self.cluster.items(),
                key=lambda item: (-item[1].failure_rate / max(item[1].mean_cost, 1e-9), item[0]),
            )
        ]
        self.global_order = {cluster: index for index, cluster in enumerate(self.ranked_clusters)}

    def new_state(self, candidate: ReplayCandidate) -> Any:
        remaining = list(candidate.cases)
        order: list[ReplayCase] = []
        uncovered = set(self.coverage)
        while remaining and uncovered:
            best = max(
                remaining,
                key=lambda case: (
                    sum(self.coverage.get(key, 0.0) for key in case.coverage if key in uncovered)
                    / max(case.cost, 1e-9),
                    self.cluster.get(case.cluster, ClusterStat()).failure_rate,
                    -case.cost,
                ),
            )
            order.append(best)
            remaining.remove(best)
            uncovered.difference_update(best.coverage)
        order.extend(
            sorted(
                remaining,
                key=lambda case: (
                    self.global_order.get(case.cluster, 10**9),
                    case.cost,
                    case.index,
                ),
            )
        )
        return {"order": order}

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        available = {case.index for case in remaining}
        for case in state["order"]:
            if case.index in available:
                return case
        return remaining[0]


class B4Policy(Policy):
    name = "B4"

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        return min(remaining, key=lambda case: case.index)


class AdaptivePolicy(Policy):
    """Bayesian mixture over dev failure profiles, with a coverage novelty term."""

    name = "Candidate"

    def __init__(self, *, k: int = 8, p_buggy_prior: float = 0.25) -> None:
        self.k = k
        self.p_buggy_prior = p_buggy_prior
        self.profiles: list[FailureProfile] = []
        self.stats: dict[str, ClusterStat] = {}
        self.p_buggy = p_buggy_prior

    def prepare(self, dev: list[ReplayCandidate]) -> None:
        self.profiles, observed_p = build_failure_profiles(dev, self.k)
        self.stats = cluster_stats(dev)
        self.p_buggy = max(0.02, min(0.80, observed_p if observed_p > 0 else self.p_buggy_prior))

    def new_state(self, candidate: ReplayCandidate) -> Any:
        if not self.profiles:
            # Cold start: use the strongest static ordering.
            order = sorted(
                candidate.cases,
                key=lambda case: (
                    -self.stats.get(case.cluster, ClusterStat()).failure_rate / max(case.cost, 1e-9),
                    case.index,
                ),
            )
            return {"mode": "static", "order": order}
        weights = [profile.weight * self.p_buggy for profile in self.profiles]
        tested: Counter[str] = Counter()
        return {"mode": "mixture", "weights": weights, "tested": tested}

    def _score(self, case: ReplayCase, state: dict[str, Any]) -> float:
        if state["mode"] == "static":
            return 1.0
        expected = 0.0
        for weight, profile in zip(state["weights"], self.profiles):
            expected += weight * profile.failure_rates.get(case.cluster, 0.0)
        novelty = 1.0 / (1.0 + state["tested"][case.cluster])
        return expected * novelty / max(case.cost, 1e-9)

    def choose(self, candidate: ReplayCandidate, state: Any, remaining: list[ReplayCase]) -> ReplayCase:
        if state["mode"] == "static":
            return state["order"][0] if state["order"] else remaining[0]
        return max(
            remaining,
            key=lambda case: (
                self._score(case, state),
                self.stats.get(case.cluster, ClusterStat()).failure_rate / max(case.cost, 1e-9),
                -case.index,
            ),
        )

    def observe(self, candidate: ReplayCandidate, state: Any, case: ReplayCase, passed: bool) -> None:
        if state["mode"] != "mixture" or not passed:
            return
        state["tested"][case.cluster] += 1
        for index, profile in enumerate(self.profiles):
            state["weights"][index] *= 1.0 - profile.failure_rates.get(case.cluster, 0.0)
        total = sum(state["weights"])
        if total <= 1e-300:
            # All profiles are implausible: treat the candidate as likely
            # correct, but keep a small floor so remaining tests are ordered
            # by their static prior.
            state["weights"] = [1e-12] * len(self.profiles)
            total = sum(state["weights"])
        state["weights"] = [weight / total for weight in state["weights"]]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    reward: int
    cost_wall: float
    cost_cpu: float
    tests_run: int
    first_failure: str
    budget_exhausted: bool


def replay_candidate(policy: Policy, candidate: ReplayCandidate, budget: float | None) -> ReplayResult:
    if policy.name == "B0":
        return ReplayResult(1, candidate.base_wall, candidate.base_cpu, 0, "", False)
    if not candidate.cases:
        return ReplayResult(1, candidate.base_wall, candidate.base_cpu, 0, "", False)
    if budget is None:
        # B4: run every available test, not budget-matched.
        spent_wall = candidate.bank_startup
        spent_cpu = candidate.bank_startup_cpu
        tests = 0
        for case in candidate.cases:
            spent_wall += case.cost
            spent_cpu += case.cpu
            tests += 1
            if case.status != "pass":
                return ReplayResult(
                    0,
                    candidate.base_wall + spent_wall,
                    candidate.base_cpu + spent_cpu,
                    tests,
                    case.failure_kind,
                    False,
                )
        return ReplayResult(
            1,
            candidate.base_wall + spent_wall,
            candidate.base_cpu + spent_cpu,
            tests,
            "",
            False,
        )

    minimum_case = min(case.cost for case in candidate.cases)
    if budget < candidate.bank_startup + minimum_case:
        return ReplayResult(1, candidate.base_wall, candidate.base_cpu, 0, "", True)

    spent_wall = candidate.bank_startup
    spent_cpu = candidate.bank_startup_cpu
    remaining = list(candidate.cases)
    state = policy.new_state(candidate)
    tests = 0
    while remaining:
        case = policy.choose(candidate, state, remaining)
        if spent_wall + case.cost > budget:
            return ReplayResult(
                1,
                candidate.base_wall + spent_wall,
                candidate.base_cpu + spent_cpu,
                tests,
                "",
                True,
            )
        spent_wall += case.cost
        spent_cpu += case.cpu
        remaining.remove(case)
        tests += 1
        if case.status != "pass":
            return ReplayResult(
                0,
                candidate.base_wall + spent_wall,
                candidate.base_cpu + spent_cpu,
                tests,
                case.failure_kind,
                False,
            )
        policy.observe(candidate, state, case, True)
    return ReplayResult(
        1,
        candidate.base_wall + spent_wall,
        candidate.base_cpu + spent_cpu,
        tests,
        "",
        False,
    )


def policies_for(seed: int, config: dict[str, Any]) -> list[Policy]:
    profile_cfg = config.get("profiles", {})
    return [
        B0Policy(),
        B1Policy(seed),
        B2Policy(),
        B3Policy(),
        AdaptivePolicy(
            k=int(profile_cfg.get("k", 8)),
            p_buggy_prior=float(profile_cfg.get("p_buggy_prior", 0.25)),
        ),
        B4Policy(),
    ]


def prepare_policies(policies: list[Policy], dev: list[ReplayCandidate]) -> None:
    for policy in policies:
        policy.prepare(dev)
