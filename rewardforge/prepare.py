"""Build the frozen, auditable task/candidate corpus.

Responsibilities:
  * load the pinned HumanEval+/MBPP+ and base-test sources;
  * split each enhanced suite into a candidate bank and private audit cases;
  * validate every selected test on the canonical solution twice;
  * write public task files (bank only) and private audit files;
  * build the natural candidate table from pinned EvalPlus release artifacts.

The policy only ever reads ``tasks_public.jsonl``.  Audit cases and graders are
written under ``data/private/`` and are loaded by the audit supervisor only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .codec import encode_value
from .config import Paths, load_config, stable_key, stable_seed
from .download import download_everything
from .testspec import (
    TestParseError,
    base_test_program,
    build_canonical_code,
    evaluate_checker,
    execute_candidate_code,
    extract_base_call_args,
    extract_features,
    infer_mbpp_entry_point,
    load_checker,
    parse_plus_suite,
    time_limit,
    TimeoutError_,
    OVERSIZED,
)


MIN_CASES_PER_SPLIT = 8
CASE_VALIDATION_TIMEOUT_S = 2.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _normalise_candidate_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    stripped = code.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:python)?\s*\n(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1)
    return code.strip() + "\n"


def _base_program(benchmark: str, base_test_code: str, entry_point: str) -> str:
    return base_test_program(benchmark, base_test_code, entry_point)


def _encode_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve tuples/sets/inf through JSON without changing comparison semantics."""
    return [
        {**case, "args": encode_value(case["args"]), "expected": encode_value(case["expected"])}
        for case in cases
    ]


def _canonical_base_ok(canonical_code: str, entry_point: str, base_program: str) -> tuple[bool, str]:
    namespace: dict[str, Any] = {"__name__": "rewardforge_canonical_base"}
    try:
        with time_limit(CASE_VALIDATION_TIMEOUT_S * 3):
            exec(compile(canonical_code, "<canonical>", "exec"), namespace)  # noqa: S102 - trusted
            if not callable(namespace.get(entry_point)):
                return False, "entry_point_missing"
            exec(compile(base_program, "<base-test>", "exec"), namespace)  # noqa: S102 - trusted
        return True, ""
    except AssertionError:
        return False, "base_assertion_failed"
    except TimeoutError_:
        return False, "base_timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _validate_selected_cases(
    canonical_code: str,
    entry_point: str,
    suite: Any,
    checker: Any,
    *,
    bank_cap: int,
    audit_cap: int,
    seed: int,
    benchmark: str,
    task_id: str,
    base_args: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select valid, non-base-overlapping cases and split them bank/audit."""

    pool: list[dict[str, Any]] = []
    seen_args: set[str] = set()
    overlap = 0
    oversized = 0
    for source_index, (args, expected, atol) in enumerate(zip(suite.inputs, suite.expected, suite.atols)):
        if expected is OVERSIZED:
            oversized += 1
            continue
        key = stable_key(encode_value(tuple(args)))
        if key in base_args:
            overlap += 1
            continue
        if key in seen_args:
            continue
        seen_args.add(key)
        pool.append(
            {
                "source_index": source_index,
                "args": args,
                "expected": expected,
                "atol": atol,
                "features": extract_features(list(args), expected),
            }
        )

    rng = random.Random(stable_seed(seed, benchmark, task_id, "bank-audit"))
    rng.shuffle(pool)

    function, _ = execute_candidate_code(canonical_code, entry_point, timeout_s=5.0)
    bank: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    invalid = 0
    duplicate_base = overlap
    for case in pool:
        if len(bank) >= bank_cap and len(audit) >= audit_cap:
            break
        accepted = True
        for _ in range(2):
            try:
                with time_limit(CASE_VALIDATION_TIMEOUT_S):
                    output = function(*case["args"])
                passed, kind, detail = evaluate_checker(
                    checker, output, case["args"], case["expected"], case["source_index"], case["atol"]
                )
                if not passed:
                    accepted = False
                    break
            except TimeoutError_:
                accepted = False
                break
            except Exception:
                accepted = False
                break
        if not accepted:
            invalid += 1
            continue
        # Alternate assignment, filling whichever side still needs cases.
        target = bank if len(bank) <= len(audit) else audit
        if len(target) >= (bank_cap if target is bank else audit_cap):
            target = audit if target is bank else bank
        if len(target) >= (bank_cap if target is bank else audit_cap):
            continue
        target.append(case)

    stats = {
        "pool_after_dedup": len(pool),
        "base_overlap": duplicate_base,
        "invalid_quarantined": invalid,
        "bank_cases": len(bank),
        "audit_cases": len(audit),
        "source_cases": suite.n_source_cases,
        "checker_kind": suite.checker_kind,
        "oversized_quarantined": oversized,
    }
    return bank, audit, stats


def _prepare_humaneval(paths: Paths, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plus = _read_jsonl(paths.raw / "humanevalplus.jsonl")
    with gzip.open(paths.raw / "humaneval.jsonl.gz", "rt", encoding="utf-8") as handle:
        base = {row["task_id"]: row for row in (json.loads(line) for line in handle if line.strip())}

    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    caps = config["benchmarks"]["humaneval"]
    for row in plus:
        task_id = row["task_id"]
        entry_point = row["entry_point"]
        canonical_code = build_canonical_code("humaneval", row["prompt"], row["canonical_solution"])
        base_row = base.get(task_id)
        if base_row is None:
            rejected.append({"task_id": task_id, "reason": "missing_original_base_test"})
            continue
        base_code = base_row["test"]
        program = _base_program("humaneval", base_code, entry_point)
        ok, reason = _canonical_base_ok(canonical_code, entry_point, program)
        if not ok:
            rejected.append({"task_id": task_id, "reason": f"base_test_invalid:{reason}"})
            continue
        try:
            suite = parse_plus_suite(row["test"], entry_point)
            checker = load_checker(suite.assertion_source)
        except Exception as exc:
            rejected.append({"task_id": task_id, "reason": f"test_parse_error:{exc}"})
            continue
        base_args = {
            stable_key(encode_value(args))
            for args in extract_base_call_args(base_code, {"candidate", entry_point})
        }
        bank, audit, stats = _validate_selected_cases(
            canonical_code,
            entry_point,
            suite,
            checker,
            bank_cap=int(caps["bank_cases"]),
            audit_cap=int(caps["audit_cases"]),
            seed=int(config["seed"]),
            benchmark="humaneval",
            task_id=task_id,
            base_args=base_args,
        )
        if len(bank) < MIN_CASES_PER_SPLIT or len(audit) < MIN_CASES_PER_SPLIT:
            rejected.append({"task_id": task_id, "reason": f"insufficient_valid_cases:{stats}"})
            continue
        tasks.append(
            {
                "task_id": task_id,
                "benchmark": "humaneval",
                "source_task_id": int(task_id.split("/")[-1]),
                "prompt": row["prompt"],
                "entry_point": entry_point,
                "canonical_code": canonical_code,
                "base_test_code": program,
                "bank_cases": bank,
                "audit_cases": audit,
                "checker_source": suite.assertion_source,
                "stats": stats,
            }
        )
    return tasks, rejected


def _prepare_mbpp(paths: Paths, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.read_parquet(paths.raw / "mbppplus.parquet")
    tasks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    caps = config["benchmarks"]["mbpp"]
    for row in frame.to_dict("records"):
        source_id = int(row["task_id"])
        task_id = f"Mbpp/{source_id}"
        test_list = list(row["test_list"])
        test_imports = list(row["test_imports"])
        entry_point = infer_mbpp_entry_point(row["code"], test_list)
        if not entry_point:
            rejected.append({"task_id": task_id, "reason": "missing_entry_point"})
            continue
        canonical_code = row["code"].strip() + "\n"
        base_code = "\n".join([*test_imports, *test_list])
        program = _base_program("mbpp", base_code, entry_point)
        ok, reason = _canonical_base_ok(canonical_code, entry_point, program)
        if not ok:
            rejected.append({"task_id": task_id, "reason": f"base_test_invalid:{reason}"})
            continue
        try:
            suite = parse_plus_suite(row["test"], entry_point)
            checker = load_checker(suite.assertion_source)
        except Exception as exc:
            rejected.append({"task_id": task_id, "reason": f"test_parse_error:{exc}"})
            continue
        base_args = {
            stable_key(encode_value(args))
            for args in extract_base_call_args(base_code, {entry_point})
        }
        bank, audit, stats = _validate_selected_cases(
            canonical_code,
            entry_point,
            suite,
            checker,
            bank_cap=int(caps["bank_cases"]),
            audit_cap=int(caps["audit_cases"]),
            seed=int(config["seed"]),
            benchmark="mbpp",
            task_id=task_id,
            base_args=base_args,
        )
        if len(bank) < MIN_CASES_PER_SPLIT or len(audit) < MIN_CASES_PER_SPLIT:
            rejected.append({"task_id": task_id, "reason": f"insufficient_valid_cases:{stats}"})
            continue
        tasks.append(
            {
                "task_id": task_id,
                "benchmark": "mbpp",
                "source_task_id": source_id,
                "prompt": row["prompt"],
                "entry_point": entry_point,
                "canonical_code": canonical_code,
                "base_test_code": program,
                "bank_cases": bank,
                "audit_cases": audit,
                "checker_source": suite.assertion_source,
                "stats": stats,
            }
        )
    return tasks, rejected


def _candidate_index(extracted: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, list[Path]]]:
    """(benchmark, generator) -> task folder name -> candidate .py files."""

    index: dict[tuple[str, str], dict[str, list[Path]]] = {}
    for entry in extracted:
        benchmark = entry["benchmark"]
        generator = entry["generator"]
        root = Path(entry["dir"])
        mapping: dict[str, list[Path]] = defaultdict(list)
        for path in root.rglob("*.py"):
            # Release layout: <root>/<model>/<HumanEval_0|Mbpp_2>/<sample>.py
            folder = path.parent.name
            mapping[folder].append(path)
        for folder in mapping:
            mapping[folder].sort()
        index[(benchmark, generator)] = mapping
    return index


def _task_folder(task_id: str) -> str:
    benchmark, number = task_id.split("/")
    return f"HumanEval_{number}" if benchmark == "HumanEval" else f"Mbpp_{number}"


def _build_candidates(
    selected_tasks: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    index = _candidate_index(extracted)
    generator_meta = {
        (benchmark, generator["name"]): generator
        for benchmark, generators in config["generators"].items()
        for generator in generators
    }
    rows: list[dict[str, Any]] = []
    for task in selected_tasks:
        benchmark = task["benchmark"]
        folder = _task_folder(task["task_id"])
        for (bm, generator_name), mapping in index.items():
            if bm != benchmark:
                continue
            files = mapping.get(folder, [])
            meta = generator_meta[(benchmark, generator_name)]
            for sample_index, path in enumerate(files):
                code = _normalise_candidate_code(path.read_text(encoding="utf-8", errors="replace"))
                if not code.strip():
                    continue
                rows.append(
                    {
                        "candidate_id": f"{benchmark}:{generator_name}:{task['task_id']}:{sample_index}",
                        "task_id": task["task_id"],
                        "benchmark": benchmark,
                        "split": task["split"],
                        "generator": generator_name,
                        "family": meta["family"],
                        "sample_index": sample_index,
                        "code": code,
                        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                        "source_path": str(path.relative_to(Paths().root)),
                    }
                )
    return rows


def prepare(config_path: str | Path | None = None, *, force_download: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    paths = Paths()
    paths.ensure()
    download_info = download_everything(config_path, force=force_download)
    extracted = download_info["extracted"]

    humaneval_tasks, humaneval_rejected = _prepare_humaneval(paths, config)
    mbpp_tasks, mbpp_rejected = _prepare_mbpp(paths, config)
    all_tasks = humaneval_tasks + mbpp_tasks
    rejected = humaneval_rejected + mbpp_rejected

    # Deterministic task split, disjoint by task ID.
    selected: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    for benchmark in ("humaneval", "mbpp"):
        candidates = [task for task in all_tasks if task["benchmark"] == benchmark]
        candidates.sort(key=lambda task: stable_key(config["seed"], benchmark, task["task_id"]))
        caps = config["benchmarks"][benchmark]
        dev_n = min(int(caps["dev_tasks"]), len(candidates))
        eval_n = min(int(caps["eval_tasks"]), max(0, len(candidates) - dev_n))
        dev = candidates[:dev_n]
        eval_ = candidates[dev_n : dev_n + eval_n]
        for task in dev:
            task["split"] = "dev"
            selected.append(task)
        for task in eval_:
            task["split"] = "eval"
            selected.append(task)
        split_manifest[benchmark] = {
            "dev": [task["task_id"] for task in dev],
            "eval": [task["task_id"] for task in eval_],
            "unused_valid": [task["task_id"] for task in candidates[dev_n + eval_n :]],
            "available_valid": len(candidates),
            "requested": {"dev": caps["dev_tasks"], "eval": caps["eval_tasks"]},
        }

    candidate_rows = _build_candidates(selected, extracted, config)
    candidate_counts = Counter((row["split"], row["benchmark"], row["generator"]) for row in candidate_rows)

    # Public tasks contain bank tests only; audit cases/graders are private.
    public_tasks = []
    audit_tasks = []
    grader_tasks = []
    reference_tasks = []
    for task in selected:
        public_tasks.append(
            {
                "task_id": task["task_id"],
                "benchmark": task["benchmark"],
                "split": task["split"],
                "source_task_id": task["source_task_id"],
                "prompt": task["prompt"],
                "entry_point": task["entry_point"],
                "base_test_code": task["base_test_code"],
                "bank_cases": _encode_cases(task["bank_cases"]),
                "stats": task["stats"],
            }
        )
        audit_tasks.append({"task_id": task["task_id"], "audit_cases": _encode_cases(task["audit_cases"])})
        grader_tasks.append({"task_id": task["task_id"], "checker_source": task["checker_source"]})
        reference_tasks.append(
            {
                "task_id": task["task_id"],
                "canonical_code": task["canonical_code"],
                "prompt": task["prompt"],
            }
        )

    paths.processed.mkdir(parents=True, exist_ok=True)
    paths.private.mkdir(parents=True, exist_ok=True)
    n_public = _write_jsonl(paths.processed / "tasks_public.jsonl", public_tasks)
    n_audit = _write_jsonl(paths.private / "audit_cases.jsonl", audit_tasks)
    n_graders = _write_jsonl(paths.private / "graders.jsonl", grader_tasks)
    n_reference = _write_jsonl(paths.private / "reference.jsonl", reference_tasks)
    n_candidates = _write_jsonl(paths.processed / "candidates.jsonl", candidate_rows)

    splits = {
        "seed": config["seed"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": split_manifest,
        "counts": {
            "selected_tasks": len(selected),
            "public_tasks": n_public,
            "audit_tasks": n_audit,
            "graders": n_graders,
            "reference_tasks": n_reference,
            "candidates": n_candidates,
        },
        "candidate_counts": [
            {"split": split, "benchmark": benchmark, "generator": generator, "count": count}
            for (split, benchmark, generator), count in sorted(candidate_counts.items())
        ],
        "rejected_tasks": rejected,
        "config": config,
    }
    (paths.processed / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "P1 audit corpus",
        "download": download_info["manifest"],
        "corpus": {
            "selected_tasks": len(selected),
            "dev_tasks": sum(1 for task in selected if task["split"] == "dev"),
            "eval_tasks": sum(1 for task in selected if task["split"] == "eval"),
            "candidates": len(candidate_rows),
            "bank_cases": sum(len(task["bank_cases"]) for task in selected),
            "audit_cases": sum(len(task["audit_cases"]) for task in selected),
            "rejected_tasks": rejected,
        },
        "files": {
            "tasks_public": str((paths.processed / "tasks_public.jsonl").relative_to(paths.root)),
            "candidates": str((paths.processed / "candidates.jsonl").relative_to(paths.root)),
            "audit_cases": str((paths.private / "audit_cases.jsonl").relative_to(paths.root)),
            "splits": str((paths.processed / "splits.json").relative_to(paths.root)),
        },
        "notes": [
            "Bank and audit cases are disjoint by construction.",
            "Every bank/audit case passed the canonical solution twice during preparation.",
            "Base tests were checked against the canonical solution before a task entered the corpus.",
            "The policy loader never reads data/private/.",
        ],
    }
    (paths.root / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the RewardForge P1 corpus")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.config, force_download=args.force_download)
    print(json.dumps(manifest["corpus"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
