# RewardForge — Feasibility and Phase 1 Execution Plan

## 1. Project definition

**Goal:** Improve the reliability of executable rewards for code-generation RL at a controlled verification cost.

Given a coding task, a generated candidate program, a validated pool of executable tests, and a wall-clock verification budget, choose which additional tests to run and when to stop. Return a reward and diagnostic evidence. The initial scope is Python function-completion tasks and a **binary correctness reward**. The verifier must not alter or expose private grading material to the candidate program.

**Hypothesis:** An adaptive test-selection policy that uses **only observations available before and during grading** can detect more incorrect base-test-passing solutions than strong static test subsets at the **same end-to-end verification cost**. Its effectiveness must remain measurable when the candidate distribution shifts.

**Important:** This is a *hypothesis*, not an achieved performance result. Improving tests, adaptive prioritization, and model-guided test generation have substantial prior art. The initial contribution is a repeatable, RL-specific, cost-controlled evaluation and a practical policy, if the results warrant it.

## 2. Prior art and differentiation

| Existing work | What it already addresses | Required distinction |
|---|---|---|
| [EvalPlus](https://github.com/evalplus/evalplus) | Enhanced HumanEval+/MBPP+ tests; evaluation harness | Use it as **oracle / benchmark**, not claim stronger tests are new |
| [RobustTests (2026)](https://arxiv.org/abs/2608.24135) | Faulty-code-guided test synthesis, validation, reward shaping; RL result | No claim to invent faulty-code-conditioned testing |
| [Themis (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40741) | Constraint-aware test synthesis and error-guided suite reduction | Compare against static error-informed suite selection |
| [MIST-RL (2026)](https://arxiv.org/abs/2603.01409) | Mutation-based incremental testing and reduced redundant tests | Focus on *execution-time selection and reward reliability*, not training a new test-generator model |
| [TCS (EMNLP 2026)](https://xiaobanni.github.io/projects/tcs/) | Policy-aligned adversarial test generator; code+verifier co-training | No novelty claim for adapting tests to evolving policy errors |
| [CATCH](https://github.com/THUAIS-Lab/CATCH) | Reward-hacking testbed and independent audit labels | Potential second-stage stress test, not a replacement for task correctness audit |
| [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers) | Tasksets, reward hooks, isolated grading containers | Implement a **verification/reward policy**, not another sandbox/rollout framework |

**Originality risk: medium–high.** Budgeted selection and test prioritization also have a history in software testing. There is no verified unique algorithm yet. The evidence-backed engineering case would be better reward decisions and/or lower cost versus strong baselines, as a reusable component.

## 3. First experiment (CPU-only): falsify before building

### Dataset and candidate programs

1. Start with **HumanEval+** and **MBPP+**. Pin an EvalPlus version, record dataset hashes and licenses, and inspect canonical solutions and input contracts.
2. Prefer **public, previously generated model solutions**, including EvalPlus release artifacts; include two generator families if available. Mutated programs may supplement the analysis but **cannot be the only evidence**.
3. Construct a candidate table with `(task_id, candidate_id, generator, code_hash)`.
4. Use each task's original/base tests as the **weak reward**. Independently run the held-out enhanced suite to assign a stronger **audit label**. Call this a *suite-based gold proxy*, not mathematical proof of program correctness.
5. Keep dev and evaluation **disjoint by task ID**, and report results by generator family and by problem difficulty. Avoid using evaluation-suite outcomes to train, rank, or choose tests.

### Test bank and data separation

For each development task, split additional validated test cases into: (a) a **candidate test bank** available to the reward policy, and (b) **private audit tests** withheld for final correctness assessment. The final audit set is not queried by the selection algorithm. Validate candidate-bank tests on the canonical reference; quarantine flaky, invalid, ambiguous, and non-deterministic tests.

For generalization tests, freeze the policy and evaluate on disjoint task IDs. If the method requires per-task historical failures, report separate **cold-start** and **after-observation** results. No oracle labels from the evaluation split may leak into policy training.

### Baselines (identical budget)

- `B0`: original base tests only.
- `B1`: original base tests + random validated tests, early stop on first failure.
- `B2`: original base tests + strongest static ranking selected from development data (e.g., historical failure rate / cost).
- `B3`: original base tests + development-only greedy coverage/set-cover ranking, where applicable.
- `B4`: all candidate-bank tests (cost/quality reference, *not* a budget-matched comparison).
- `Candidate`: sequential, adaptive selector using only legal observed features, runtime estimates, and prior non-private failure history. Begin with a simple deterministic heuristic; use bandits/learned policies **only if a simple policy cannot explain a gain**.

All methods use the **same measured CPU verification budget**, including subprocess/container overhead. Vary budgets and always include the full time-vs-quality frontier. In the initial offline replay, charge the *measured* per-test cost; confirm top results through actual sandbox execution to detect replay artefacts.

### Required metrics

- **False-positive reward rate:** fraction of independently audited incorrect programs that receive positive reward. Provide denominators.
- **True-positive acceptance:** fraction of audited correct programs retained as correct (detect false rejection).
- **Precision / reward disagreement:** overall and by task, generator, and error category.
- **Cost:** p50/p95/p99 wall-clock grading time per candidate, CPU-seconds, test executions, and sandbox startup overhead.
- **Value at fixed cost:** quality-versus-runtime frontier against **B1–B3**.
- **Statistical uncertainty:** paired comparisons, bootstrap confidence intervals **clustered by task**, at multiple budgets.

### Security and correctness

Execute candidate code in a resource-limited **isolated sandbox**, with no network, no access to reference implementation or private tests, and explicit time/memory/process/output limits. Do not run untrusted candidate code in the evaluator's Python process. Distinguish candidate failure (reward 0) from harness infrastructure failure (do not silently label as incorrect).

## 5. Go/no-go rules after the pilot

**Proceed to build the full project only if ALL hold:**

1. There is a meaningful real-world signal: at least ~30 **naturally generated** base-pass/audit-fail candidates on the held-out evaluation set, from multiple tasks. If the sample is smaller, expand the corpus rather than claim a percentage improvement.
2. The proposed adaptive strategy reduces false positives by a **pre-registered target of ≥30% relative to the best B1–B3** at similar p50/p95 wall-clock cost. The task-clustered confidence interval should favor it. This is a success **criterion**, not an expected finding.
3. No material increase in false rejection for independently audited correct programs, and no unvalidated tests accepted.
4. Improvement remains on disjoint tasks/generators; it is not simply memorization of failing tests for training tasks.
5. Real runtime overhead is small enough that *using all tests* is not trivially more economical. If sandbox startup dominates or exhaustive checking is cheap, adaptive selection may have no practical case.

**If the adaptive selector fails but the audit exposes a serious reproducible verifier defect:** consider a narrow verifier correctness contribution. Otherwise, **NO-GO**. Do not build an RL wrapper around a weak result.

## 6. Implementation phases after the first gate

| Phase | Concrete deliverable | Completion condition |
|---|---|---|
| P1 — Audit harness | Pinned dataset, candidate corpus, task/test split, sandbox, result matrix | Repeatable base-vs-private-audit confusion matrix and runtime stats |
| P2 — Budgeted verifier | Fixed-budget baselines, deterministic adaptive selector, traces | Correctness, soundness, cost, task-level uncertainty on held-out tasks |
| P3 — RL integration | `verifiers.v1` task and `@vf.reward` plugin, isolated scoring, trace diagnostics | Works with existing rollout data, no leakage or solver/grader cross-contamination |
| P4 — RL evidence | Baseline vs improved-reward small-model training runs | Equal training-token/compute comparison; held-out correctness plus reward-noise trends |
| P5 — Publication | Repro scripts, data cards, evaluation charts, code and write-up; optional upstream PR | Another engineer can reproduce the result |

**The RL phase is required for a strong claim that RewardForge improves learning**, but a $50 initial GPU budget may support only a *pilot*, not multi-seed statistically credible training. Full training is contingent on verified P1/P2 value and transparent resource accounting.

## 7. What to do first

1. Pin EvalPlus and choose 30–50 development tasks and 30–50 untouched evaluation tasks; collect natural candidate solutions with both passing and failing cases.
2. Build `rewardforge/audit.py`: run base tests and private audit tests, writing a task × candidate × test result table with wall times and failure modes.
3. Establish the rate of **base-pass / private-audit-fail** on natural candidates, together with real test-execution and container costs.
4. Only then implement and compare adaptive selection against B1–B3.

**First artifact to ship:** `audit_summary.json`, `cases.parquet` (or CSV), a reproducible audit command, and a chart of false-positive rewards at each cost budget. **No GPU necessary.**

## 8. Overall research verdict

**Research feasibility:** PASS — existing data and grading frameworks support a zero-GPU pilot.  
**Novelty:** NOT YET PROVEN — substantial adjacent research.  
**Practical impact:** PLAUSIBLE BUT UNMEASURED — could reduce bad RL rewards; benefit depends on real task/test cost and failure patterns.  
**Resume replacement:** HOLD — do not replace MiniServe until the CPU gates pass and, ideally, training results show downstream value.  
**Next action:** Execute P1, beginning with a reproducible baseline reward audit.
