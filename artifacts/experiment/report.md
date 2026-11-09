# RewardForge — first experiment report

This is a CPU-only falsification experiment. It is **not** an RL training run.
The policy is compared at the same measured wall-clock budget, using the
P1 audit's per-test costs and sandbox startup.

## Corpus

- Development candidates: 275
- Held-out evaluation candidates: 279 (32 audited incorrect, 247 audited correct)
- Evaluation tasks are disjoint from development tasks by task ID.
- Base tests are the weak reward; private enhanced-suite audit is the label.
- Base-pass/audit-fail rate: 12.5% overall (69/554); 11.5% on held-out eval (32/279).
- By benchmark: HumanEval+ 7.5% (22/294), MBPP+ 18.1% (47/260).

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| 1. Natural signal | PASS | 32 base-pass/audit-fail on eval across 27 tasks (target >=30) |
| 2. Adaptive gain | FAIL | best observed relative FPR reduction 0.000 at 0.0375 s (target >=0.30) |
| 3. False rejection | PASS | candidate 0.004 vs B4 0.004 |
| 4. Disjoint generalization | FAIL | generator holdout gain: False |
| 5. Cost realism | FAIL | median test time 0.002 s is 1.7% of total; startup is 94.0% of bank stage |

**Overall: NO-GO (first-experiment gate).**

## What the numbers say

- B0 (base only) has FPR 1.000, p50 75.9 ms.
- B4 (all bank tests) has FPR 0.000, p50 110.8 ms, p95 157.0 ms.
- At the best-test-running budget (37.5 ms), B2 has FPR 0.125; the candidate has 0.125. No budget produced a positive relative reduction; at 33–35 ms the candidate was worse.
- Median bank test time is 1.86 ms; median bank startup is 32.75 ms. The measured cost structure leaves almost no room for test selection to matter.
- Secondary metrics: at 33 ms the candidate has TPR-accept 1.000 (false-rejection 0.000, precision 0.936); at 60 ms TPR-accept 0.996 (false-rejection 0.004, precision 0.988). B4 has TPR-accept 0.996, false-rejection 0.004, precision 1.000. The candidate does not improve the correctness trade-off; it only spends less before converging to the same residual errors.

## Representative points

| setup | p50 wall (ms) | p95 wall (ms) | FPR | false positives / incorrect | candidate vs best baseline |
|---|---:|---:|---:|---:|---:|
| B0: base only | 75.9 | 84.8 | 1.000 | 32/32 | n/a |
| B2 static @ 37.5 ms | 110.6 | 120.2 | 0.125 | 4/32 | n/a |
| Candidate @ 37.5 ms | 110.6 | 120.2 | 0.125 | 4/32 | 0.000 |
| B3 static @ 33.0 ms | 107.7 | 110.6 | 0.438 | 14/32 | n/a |
| Candidate @ 33.0 ms | 107.7 | 110.6 | 0.531 | 17/32 | -0.214 |
| B2 static @ 60 ms | 110.8 | 134.4 | 0.094 | 3/32 | n/a |
| Candidate @ 60 ms | 110.8 | 133.9 | 0.094 | 3/32 | 0.000 |
| B2 static @ 160 ms | 110.8 | 150.0 | 0.094 | 3/32 | n/a |
| Candidate @ 160 ms | 110.8 | 153.3 | 0.094 | 3/32 | 0.000 |
| B4: all bank tests | 110.8 | 157.0 | 0.000 | 0/32 | n/a |

*Counts are exact integer counts from the replay; see `frontier.csv`. The key point:
at essentially the same p50 cost, running all tests reaches FPR 0 while budget-matched
policies still leave 3 false positives from slow-tail tests.

At 37.5 ms the candidate FPR is 0.125 (task-clustered 95% CI 0.026–0.265) and B2 is 0.125 (CI 0.028–0.263). The paired difference (candidate - B2) is 0.000 to 0.000 FPR. At 33.0 ms the candidate FPR is 0.531 versus 0.438 for B3, i.e. the candidate is worse. At 60.0 ms the paired difference is 0.000 with a zero-width interval.

## Breakdown at the best-test-running budget

| group | n | incorrect | best baseline | best FPR | candidate FPR |
|---|---:|---:|---|---:|---:|
| humaneval | 148 | 11 | B2 | 0.273 | 0.273 |
| mbpp | 131 | 21 | B2 | 0.048 | 0.048 |
| chatgpt | 58 | 6 | B2 | 0.167 | 0.167 |
| codegen-2b | 21 | 2 | B2 | 0.500 | 0.500 |
| deepseek-coder-6.7b-base | 61 | 13 | B2 | 0.000 | 0.000 |
| gpt-4-1106-preview | 139 | 11 | B2 | 0.182 | 0.182 |
| easy | 57 | 2 | B2 | 0.000 | 0.000 |
| hard | 26 | 6 | B2 | 0.500 | 0.500 |
| medium | 196 | 24 | B2 | 0.042 | 0.042 |

Difficulty is a descriptive proxy (task base-pass rate over all candidates), not a policy input.

### Generator holdout

| held-out generator | eval n | incorrect | max relative reduction | budget |
|---|---:|---:|---:|---:|
| chatgpt | 58 | 6 | 0.000 | 0.0325 s |
| codegen-2b | 21 | 2 | 0.000 | 0.032 s |
| deepseek-coder-6.7b-base | 61 | 13 | 0.000 | 0.032 s |
| gpt-4-1106-preview | 139 | 11 | 0.000 | 0.032 s |

## Budget frontier

| budget (ms) | best baseline | best FPR | candidate FPR | relative reduction | candidate p50 (ms) |
|---:|---|---:|---:|---:|---:|
| 32.0 | B2 | 0.969 | 0.969 | 0.000 | 75.9 |
| 32.5 | B2 | 0.844 | 0.844 | 0.000 | 76.6 |
| 33.0 | B3 | 0.438 | 0.531 | -0.214 | 107.7 |
| 33.5 | B3 | 0.250 | 0.344 | -0.375 | 108.7 |
| 34.0 | B1 | 0.219 | 0.250 | -0.143 | 109.2 |
| 35.0 | B1 | 0.188 | 0.219 | -0.167 | 109.9 |
| 37.5 | B2 | 0.125 | 0.125 | 0.000 | 110.6 |
| 40.0 | B2 | 0.125 | 0.125 | 0.000 | 110.7 |
| 50.0 | B2 | 0.125 | 0.125 | 0.000 | 110.7 |
| 60.0 | B2 | 0.094 | 0.094 | 0.000 | 110.8 |
| 80.0 | B2 | 0.094 | 0.094 | 0.000 | 110.8 |
| 120.0 | B2 | 0.094 | 0.094 | 0.000 | 110.8 |
| 160.0 | B2 | 0.094 | 0.094 | 0.000 | 110.8 |
| 250.0 | B2 | 0.094 | 0.094 | 0.000 | 110.8 |

## Interpretation

- Gate 1 passes narrowly: 32 held-out base-pass/audit-fail candidates across 27 tasks.
- The adaptive selector does not produce the pre-registered 30% relative FPR reduction. It ties the best static baseline at 37.5 ms, is 14–38% worse at 33–35 ms, and is identical at most larger budgets. No budget produced a positive relative reduction.
- The dominant cost is process/sandbox startup, not test execution. For the median candidate, running all bank tests costs about 1.86 ms of test time on top of 32.75 ms of startup, while the whole grading call is 110.9 ms. Selection cannot save enough to matter at this scale.
- This is a **NO-GO for the adaptive-selection premise as tested**. It is not evidence that the audit harness is useless: the audit itself found a reproducible 12.5% base-pass/audit-fail rate among base-passing candidates (11.5% on eval).

## Limitations

- Audit labels come from a held-out subset of the enhanced suite, not a proof of correctness.
- Bank and audit tests are sampled from EvalPlus suites; detection power is finite.
- All 966 candidates were audited for completeness, but the policy comparison uses the 554 base-passing candidates. Audit cases exclude base-overlap inputs, so the 17/412 base-fail/audit-pass candidates bound base-test false rejection only from above; they are not proof that a base test was wrong.
- Costs are measured on one macOS host with one worker per suite; a per-test-container design would have different cost structure and is a separate experiment.
- The candidate corpus is public model output, not a fresh RL policy distribution.
- 32 evaluation false positives is just above the pre-registered minimum; the confidence intervals are wide and the result should not be over-interpreted as a precise rate.

## Artifacts

- `artifacts/audit/audit_summary.json` — P1 counts, costs, gate 1/5 evidence.
- `artifacts/audit/cases.csv` — task × candidate × test result table.
- `artifacts/audit/candidates.csv` — candidate-level base/bank/audit outcomes.
- `artifacts/experiment/metrics.json` — all replay metrics, CIs, gate status.
- `artifacts/experiment/frontier.csv` — quality/cost frontier.
- `artifacts/experiment/false_positive_frontier.png` — false-positive rewards by budget/cost.
- `artifacts/experiment/cost_breakdown.png` — where grading time goes.
