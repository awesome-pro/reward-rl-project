"""Command-line entry point for the RewardForge first experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rewardforge",
        description="RewardForge P1 audit + P2 budgeted adaptive-selector experiment (CPU only)",
    )
    parser.add_argument("--config", default=None, help="path to a JSON config")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", parents=[common], help="download pinned data and build the frozen corpus")
    prepare_parser.add_argument("--force-download", action="store_true")

    audit_parser = subparsers.add_parser("audit", parents=[common], help="run the P1 base/bank/private-audit matrix")
    audit_parser.add_argument("--limit", type=int, default=None)
    audit_parser.add_argument("--task", default=None)
    audit_parser.add_argument("--no-resume", action="store_true")

    subparsers.add_parser("experiment", parents=[common], help="run the P2 budgeted replay and write the frontier")
    subparsers.add_parser("all", parents=[common], help="prepare, audit, then experiment")

    args = parser.parse_args(argv)
    config = args.config

    if args.command in {"prepare", "all"}:
        from .prepare import prepare

        manifest = prepare(config, force_download=getattr(args, "force_download", False))
        print(json.dumps(manifest["corpus"], indent=2))

    if args.command in {"audit", "all"}:
        from .audit import run_audit

        summary = run_audit(
            config,
            limit=getattr(args, "limit", None),
            task_filter=getattr(args, "task", None),
            resume=not getattr(args, "no_resume", False),
        )
        print(json.dumps({key: summary[key] for key in ("counts", "rates", "gate_1", "gate_5")}, indent=2))

    if args.command in {"experiment", "all"}:
        from .experiment import run_experiment

        metrics = run_experiment(config)
        print(json.dumps(metrics["overall"], indent=2))
        print(json.dumps(metrics["gate_2_adaptive_gain"], indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
