#!/usr/bin/env python3
"""Minimal, isolated candidate worker for RewardForge.

The worker is deliberately standalone (stdlib only) and is started as a fresh
process by the trusted supervisor.  It receives only the current test
arguments over stdin, invokes the candidate function, and returns the result
over stdout.  Expected values, assertions, future tests, and private audit
tests never enter this process.

Protocol: one JSON object per line.
  supervisor -> worker: {"id": 1, "args": <codec>, "kwargs": <codec>}
  worker -> supervisor: {"id": 1, "ok": true, "value": <codec>, ...}
The candidate's own stdout is redirected to stderr so it cannot corrupt the
protocol.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
import resource
import socket
import sys
import time


# ---------------------------------------------------------------------------
# Value codec (kept in sync with rewardforge/codec.py; no package imports so
# the worker can run under `python -I`).
# ---------------------------------------------------------------------------


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict) or "__rf__" not in value:
        return value
    tag = value["__rf__"]
    if tag == "float":
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[value["v"]]
    if tag == "complex":
        return complex(value["real"], value["imag"])
    if tag == "bytes":
        return base64.b64decode(value["b64"])
    if tag == "tuple":
        return tuple(decode_value(item) for item in value["v"])
    if tag == "set":
        return {decode_value(item) for item in value["v"]}
    if tag == "frozenset":
        return frozenset(decode_value(item) for item in value["v"])
    if tag == "dict":
        return {decode_value(k): decode_value(v) for k, v in value["v"]}
    if tag == "opaque":
        return value
    raise ValueError(f"unknown codec tag {tag!r}")


def encode_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__rf__": "float", "v": "nan"}
        if math.isinf(value):
            return {"__rf__": "float", "v": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, complex):
        return {"__rf__": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, bytes):
        return {"__rf__": "bytes", "b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"__rf__": "tuple", "v": [encode_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, set):
        return {"__rf__": "set", "v": _sort([encode_value(item) for item in value])}
    if isinstance(value, frozenset):
        return {"__rf__": "frozenset", "v": _sort([encode_value(item) for item in value])}
    if isinstance(value, dict):
        return {"__rf__": "dict", "v": [[encode_value(k), encode_value(v)] for k, v in value.items()]}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {"__rf__": "ndarray", "dtype": str(value.dtype), "v": [encode_value(item) for item in value.tolist()]}
        if isinstance(value, np.generic):
            return encode_value(value.item())
    except Exception:
        pass
    return {"__rf__": "opaque", "type": type(value).__qualname__, "repr": repr(value)[:500]}


def _sort(values):
    return sorted(values, key=repr)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def _apply_limits(cpu_s: int, fsize_mb: int, nofile: int, nproc: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    except Exception:
        pass
    try:
        fsize = fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
    except Exception:
        pass
    try:
        os.umask(0o077)
    except Exception:
        pass


def _block_network() -> None:
    """Best-effort network denial inside the worker.

    This is not a kernel boundary.  When the host supports macOS sandbox-exec
    the supervisor wraps this process in a profile that denies network at the
    kernel level; otherwise the supervisor records ``network=patch`` in the
    manifest.  The patch stops ordinary socket use before candidate code runs.
    """

    def blocked(*_args, **_kwargs):
        raise OSError("network access is disabled during grading")

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    socket.gethostbyname = blocked  # type: ignore[assignment]
    original_socket = socket.socket

    class GuardedSocket(original_socket):  # type: ignore[misc]
        def connect(self, *args, **kwargs):
            blocked()

        def connect_ex(self, *args, **kwargs):
            blocked()

        def sendto(self, *args, **kwargs):
            blocked()

    socket.socket = GuardedSocket  # type: ignore[assignment]


def _load_candidate(code_path: str, entry_point: str):
    spec = importlib.util.spec_from_file_location("rewardforge_candidate", code_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create module spec for candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, entry_point, None)
    if not callable(function):
        raise AttributeError(f"entry point {entry_point!r} is not callable")
    return function, module


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def serve(code_path: str, entry_point: str, load_timeout_s: float) -> int:
    protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    try:
        function, _module = _load_candidate(code_path, entry_point)
    except BaseException as exc:
        protocol_out.write(
            json.dumps(
                {
                    "ready": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "load_s": (time.perf_counter_ns() - started) / 1e9,
                }
            )
            + "\n"
        )
        protocol_out.flush()
        return 3
    protocol_out.write(
        json.dumps(
            {
                "ready": True,
                "load_s": (time.perf_counter_ns() - started) / 1e9,
                "cpu_s": (time.process_time_ns() - cpu_started) / 1e9,
            }
        )
        + "\n"
    )
    protocol_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception:
            protocol_out.write(json.dumps({"ok": False, "harness_error": "bad request json"}) + "\n")
            protocol_out.flush()
            continue
        request_id = request.get("id")
        args = decode_value(request.get("args", []))
        kwargs = decode_value(request.get("kwargs", {}))
        if not isinstance(args, (list, tuple)):
            args = [args]
        if not isinstance(kwargs, dict):
            kwargs = {}
        start = time.perf_counter_ns()
        cpu0 = time.process_time_ns()
        try:
            output = function(*args, **kwargs)
            encoded = encode_value(output)
            payload = json.dumps({"id": request_id, "ok": True, "value": encoded}, separators=(",", ":"))
            if len(payload) > 2_000_000:
                payload = json.dumps(
                    {
                        "id": request_id,
                        "ok": False,
                        "error_type": "OutputTooLarge",
                        "error": "encoded output exceeded 2 MB",
                        "wall_s": (time.perf_counter_ns() - start) / 1e9,
                        "cpu_s": (time.process_time_ns() - cpu0) / 1e9,
                    }
                )
        except BaseException as exc:
            payload = json.dumps(
                {
                    "id": request_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "wall_s": (time.perf_counter_ns() - start) / 1e9,
                    "cpu_s": (time.process_time_ns() - cpu0) / 1e9,
                }
            )
        else:
            # Include timings on success too.
            record = json.loads(payload)
            record["wall_s"] = (time.perf_counter_ns() - start) / 1e9
            record["cpu_s"] = (time.process_time_ns() - cpu0) / 1e9
            payload = json.dumps(record, separators=(",", ":"))
        protocol_out.write(payload + "\n")
        protocol_out.flush()
    return 0


def run_base(code_path: str, entry_point: str, base_path: str) -> int:
    protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    start = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    try:
        namespace = {"__name__": "rewardforge_candidate_base"}
        with open(code_path, encoding="utf-8") as handle:
            code = handle.read()
        exec(compile(code, code_path, "exec"), namespace)  # noqa: S102 - candidate process
        function = namespace.get(entry_point)
        if not callable(function):
            raise AttributeError(f"entry point {entry_point!r} is not callable")
        with open(base_path, encoding="utf-8") as handle:
            base_code = handle.read()
        # Stabilise the public base suite.  EvalPlus' plus tests are
        # deterministic, but a few original HumanEval checks use random().
        import random

        random.seed(0)
        try:
            import numpy as np

            np.random.seed(0)
        except Exception:
            pass
        exec(compile(base_code, base_path, "exec"), namespace)  # noqa: S102 - public base test
        result = {
            "ok": True,
            "wall_s": (time.perf_counter_ns() - start) / 1e9,
            "cpu_s": (time.process_time_ns() - cpu0) / 1e9,
        }
    except BaseException as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "wall_s": (time.perf_counter_ns() - start) / 1e9,
            "cpu_s": (time.process_time_ns() - cpu0) / 1e9,
        }
    protocol_out.write(json.dumps(result, separators=(",", ":")) + "\n")
    protocol_out.flush()
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["serve", "base"])
    parser.add_argument("code_path")
    parser.add_argument("entry_point")
    parser.add_argument("--base-test", default=None)
    parser.add_argument("--cpu-limit", type=int, default=12)
    parser.add_argument("--fsize-mb", type=int, default=2)
    parser.add_argument("--nofile", type=int, default=64)
    parser.add_argument("--nproc", type=int, default=0)
    parser.add_argument("--load-timeout", type=float, default=8.0)
    parser.add_argument("--network", choices=["patch", "off"], default="patch")
    args = parser.parse_args()

    _apply_limits(args.cpu_limit, args.fsize_mb, args.nofile, args.nproc)
    if args.network == "patch":
        _block_network()
    if args.mode == "serve":
        return serve(args.code_path, args.entry_point, args.load_timeout)
    if not args.base_test:
        raise SystemExit("--base-test is required in base mode")
    return run_base(args.code_path, args.entry_point, args.base_test)


if __name__ == "__main__":
    sys.exit(main())
