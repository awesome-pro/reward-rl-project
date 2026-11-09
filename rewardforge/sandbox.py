"""Trusted supervisor for candidate grading.

Design guarantees:

* candidate code runs in a separate process, never in the evaluator process;
* only the current test arguments cross the process boundary;
* expected values, test assertions, and private audit cases stay in the
  supervisor;
* each worker has CPU/file/process limits, a wall-clock timeout, and a
  best-effort network block; on hosts where ``sandbox-exec`` works it is
  additionally wrapped in a kernel sandbox;
* candidate failure, timeout, crash, and harness failure are distinct states.
"""

from __future__ import annotations

import json
import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codec import decode_value, encode_value
from .config import ROOT


WORKER = Path(__file__).with_name("_worker.py")


class WorkerTimeout(RuntimeError):
    pass


class WorkerEOF(RuntimeError):
    pass


class WorkerProtocolError(RuntimeError):
    pass


class WorkerMemoryLimit(RuntimeError):
    pass


@dataclass
class CaseOutcome:
    index: int
    status: str  # pass | fail | timeout | crash | not_run | harness_error
    failure_kind: str = ""
    detail: str = ""
    wall_s: float = 0.0
    cpu_s: float = 0.0
    source_index: int | None = None


@dataclass
class SuiteResult:
    status: str  # ok | load_error | timeout | crash | infra_error
    outcomes: list[CaseOutcome] = field(default_factory=list)
    startup_s: float = 0.0
    startup_cpu_s: float = 0.0
    load_s: float = 0.0
    backend: str = ""
    detail: str = ""

    @property
    def all_passed(self) -> bool:
        return self.status == "ok" and all(outcome.status == "pass" for outcome in self.outcomes)

    @property
    def first_failure(self) -> CaseOutcome | None:
        for outcome in self.outcomes:
            if outcome.status != "pass":
                return outcome
        return None


@dataclass
class BaseResult:
    status: str  # pass | fail | timeout | crash | load_error | infra_error
    wall_s: float = 0.0
    cpu_s: float = 0.0
    detail: str = ""
    backend: str = ""


def _probe_sandbox_exec() -> bool:
    executable = shutil.which("sandbox-exec")
    if not executable:
        return False
    command = [executable, "-p", "(version 1)(allow default)", sys.executable, "-c", "print('ok')"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except Exception:
        return False
    return result.returncode == 0 and "ok" in result.stdout


def _quote_profile_path(path: Path) -> str:
    return f'"{path}"'


def _sandbox_profile(temp_dir: Path) -> str:
    readable = [
        "/usr",
        "/System",
        "/Library",
        "/etc",
        "/private/etc",
        "/private/tmp",
        "/private/var/folders",
        "/opt/homebrew",
        str(Path(sys.prefix)),
        str(Path(sys.base_prefix)),
        str(ROOT / "rewardforge"),
        str(temp_dir),
    ]
    read_rules = " ".join(f"(subpath {_quote_profile_path(Path(path))})" for path in readable)
    write_rules = f"(subpath {_quote_profile_path(temp_dir)}) (literal \"/dev/null\")"
    return (
        "(version 1)"
        "(deny default)"
        "(allow process*)"
        "(allow signal)"
        "(allow sysctl-read)"
        "(allow mach-lookup)"
        "(allow file-read-metadata)"
        f"(allow file-read* {read_rules} (literal \"/dev/null\") (literal \"/dev/urandom\") (literal \"/dev/random\"))"
        f"(allow file-write* {write_rules})"
    )


def _rss_kb(pid: int) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=0.8,
        )
        text = result.stdout.strip()
        return int(text) if text else None
    except Exception:
        return None


class SandboxRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        mode = str(config.get("sandbox_exec", "auto")).lower()
        available = _probe_sandbox_exec()
        if mode == "on" and not available:
            raise RuntimeError("sandbox-exec was requested but is unavailable")
        self.use_sandbox_exec = available and mode in {"auto", "on"}
        self.backend = "sandbox-exec+network-deny" if self.use_sandbox_exec else "rlimit+netpatch"
        self.test_timeout_s = float(config.get("test_timeout_s", 2.0))
        self.base_timeout_s = float(config.get("base_timeout_s", 6.0))
        self.load_timeout_s = float(config.get("load_timeout_s", 8.0))
        self.cpu_limit_s = int(config.get("cpu_limit_s", 12))
        self.memory_limit_mb = int(config.get("memory_limit_mb", 1024))
        self.fsize_mb = int(config.get("file_size_limit_mb", 2))
        self.network = str(config.get("network", "patch"))

    # ------------------------------------------------------------------
    # Process plumbing
    # ------------------------------------------------------------------
    def _limits(self, cpu_limit_s: int):
        def _apply() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s + 1))
            except Exception:
                pass
            try:
                fsize = self.fsize_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except Exception:
                pass
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            except Exception:
                pass
            try:
                os.umask(0o077)
            except Exception:
                pass

        return _apply

    def _environment(self, temp_dir: Path) -> dict[str, str]:
        return {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "MPLCONFIGDIR": str(temp_dir),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
        }

    def _command(self, args: list[str], temp_dir: Path) -> list[str]:
        command = [sys.executable, "-I", str(WORKER), *args]
        if self.use_sandbox_exec:
            executable = shutil.which("sandbox-exec") or "sandbox-exec"
            return [executable, "-p", _sandbox_profile(temp_dir), *command]
        return command

    def _popen(self, args: list[str], temp_dir: Path, cpu_limit_s: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            self._command(args, temp_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=temp_dir,
            env=self._environment(temp_dir),
            start_new_session=True,
            preexec_fn=self._limits(cpu_limit_s),
            bufsize=0,
        )

    @staticmethod
    def _kill(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        try:
            process.wait(timeout=3)
        except Exception:
            pass

    def _send(self, process: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
        assert process.stdin is not None
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        process.stdin.write(data)
        process.stdin.flush()

    def _read_message(
        self,
        process: subprocess.Popen[bytes],
        timeout_s: float,
        buffer: bytearray,
    ) -> dict[str, Any]:
        assert process.stdout is not None
        fd = process.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        last_memory_check = time.monotonic()
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                if not line.strip():
                    continue
                try:
                    return json.loads(line.decode("utf-8", errors="replace"))
                except Exception as exc:
                    raise WorkerProtocolError(f"invalid worker JSON: {exc}") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout("worker response timed out")
            ready, _, _ = select.select([fd], [], [], min(0.1, remaining))
            now = time.monotonic()
            if now - last_memory_check >= 0.5:
                last_memory_check = now
                rss = _rss_kb(process.pid)
                if rss is not None and rss > self.memory_limit_mb * 1024:
                    raise WorkerMemoryLimit(f"worker RSS {rss} KB exceeded limit")
            if not ready:
                if process.poll() is not None:
                    raise WorkerEOF(f"worker exited with {process.returncode}")
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise WorkerEOF("worker closed stdout")
            buffer.extend(chunk)
            if len(buffer) > 4 * 1024 * 1024:
                raise WorkerProtocolError("worker response exceeded 4 MB")
            if process.poll() is not None and buffer.find(b"\n") < 0:
                raise WorkerEOF(f"worker exited with {process.returncode}")

    # ------------------------------------------------------------------
    # Public grading API
    # ------------------------------------------------------------------
    def run_suite(
        self,
        code: str,
        entry_point: str,
        cases: list[dict[str, Any]],
        checker: Any,
        *,
        stop_on_failure: bool = False,
        timeout_s: float | None = None,
    ) -> SuiteResult:
        """Run one candidate against a sequence of cases.

        ``cases`` must already have decoded ``args``/``expected`` fields.
        The candidate only receives ``args``.
        """

        temp_dir = Path(tempfile.mkdtemp(prefix="rewardforge_"))
        code_path = temp_dir / "candidate.py"
        code_path.write_text(code, encoding="utf-8")
        test_timeout = float(timeout_s if timeout_s is not None else self.test_timeout_s)
        cpu_limit = min(900, max(self.cpu_limit_s, len(cases) * int(test_timeout + 1) + 15))
        process: subprocess.Popen[bytes] | None = None
        outcomes = [
            CaseOutcome(
                index=index,
                status="not_run",
                failure_kind="not_run",
                source_index=cases[index].get("source_index", index),
            )
            for index in range(len(cases))
        ]
        startup_s = 0.0
        startup_cpu_s = 0.0
        load_s = 0.0
        buffer = bytearray()
        try:
            process = self._popen(
                [
                    "serve",
                    str(code_path),
                    entry_point,
                    "--cpu-limit",
                    str(cpu_limit),
                    "--fsize-mb",
                    str(self.fsize_mb),
                    "--network",
                    self.network,
                ],
                temp_dir,
                cpu_limit,
            )
            started = time.monotonic()
            try:
                handshake = self._read_message(process, self.load_timeout_s, buffer)
            except WorkerTimeout:
                self._kill(process)
                return SuiteResult("timeout", outcomes, startup_s, startup_cpu_s, load_s, self.backend, "load timeout")
            except (WorkerEOF, WorkerProtocolError, WorkerMemoryLimit) as exc:
                self._kill(process)
                return SuiteResult("infra_error", outcomes, startup_s, startup_cpu_s, load_s, self.backend, str(exc))
            startup_s = time.monotonic() - started
            load_s = float(handshake.get("load_s", 0.0))
            startup_cpu_s = float(handshake.get("cpu_s", 0.0))
            if not handshake.get("ready"):
                self._kill(process)
                return SuiteResult(
                    "load_error",
                    outcomes,
                    startup_s,
                    startup_cpu_s,
                    load_s,
                    self.backend,
                    f"{handshake.get('error_type', 'LoadError')}: {handshake.get('error', '')}",
                )
            for index, case in enumerate(cases):
                request = {
                    "id": index,
                    "args": encode_value(case["args"]),
                    "kwargs": encode_value(case.get("kwargs", {})),
                }
                tests_started = time.monotonic()
                try:
                    self._send(process, request)
                except (BrokenPipeError, OSError):
                    outcomes[index] = CaseOutcome(index, "crash", "worker_pipe_closed", "", 0.0, 0.0, case.get("source_index", index))
                    break
                try:
                    response = self._read_message(process, test_timeout, buffer)
                except WorkerTimeout:
                    self._kill(process)
                    outcomes[index] = CaseOutcome(index, "timeout", "timeout", "wall-clock limit", test_timeout, 0.0, case.get("source_index", index))
                    break
                except WorkerMemoryLimit:
                    self._kill(process)
                    outcomes[index] = CaseOutcome(index, "crash", "memory_limit", "memory limit", time.monotonic() - tests_started, 0.0, case.get("source_index", index))
                    break
                except (WorkerEOF, WorkerProtocolError) as exc:
                    self._kill(process)
                    outcomes[index] = CaseOutcome(index, "crash", "worker_crash", str(exc), time.monotonic() - tests_started, 0.0, case.get("source_index", index))
                    break
                wall = time.monotonic() - tests_started
                if response.get("id") != index:
                    outcomes[index] = CaseOutcome(index, "harness_error", "id_mismatch", str(response)[:300], wall, 0.0, case.get("source_index", index))
                    break
                cpu = float(response.get("cpu_s", 0.0))
                if response.get("harness_error"):
                    outcomes[index] = CaseOutcome(index, "harness_error", "protocol", response["harness_error"], wall, cpu, case.get("source_index", index))
                    break
                if not response.get("ok"):
                    kind = str(response.get("error_type", "Exception"))
                    detail = str(response.get("error", ""))[:500]
                    outcomes[index] = CaseOutcome(index, "fail", f"exception:{kind}", detail, wall, cpu, case.get("source_index", index))
                else:
                    output = decode_value(response.get("value"))
                    passed, kind, detail = self._evaluate(checker, output, case, index)
                    outcomes[index] = CaseOutcome(
                        index,
                        "pass" if passed else "fail",
                        kind,
                        detail,
                        wall,
                        cpu,
                        case.get("source_index", index),
                    )
                if stop_on_failure and outcomes[index].status != "pass":
                    break
            try:
                if process.stdin:
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=2)
            except Exception:
                self._kill(process)
            status = "ok"
            if any(outcome.status in {"timeout"} for outcome in outcomes):
                status = "timeout"
            elif any(outcome.status in {"crash"} for outcome in outcomes):
                status = "crash"
            elif any(outcome.status == "harness_error" for outcome in outcomes):
                status = "infra_error"
            return SuiteResult(status, outcomes, startup_s, startup_cpu_s, load_s, self.backend)
        finally:
            if process is not None and process.poll() is None:
                self._kill(process)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def run_base(self, code: str, entry_point: str, base_program: str) -> BaseResult:
        temp_dir = Path(tempfile.mkdtemp(prefix="rewardforge_base_"))
        code_path = temp_dir / "candidate.py"
        base_path = temp_dir / "base_test.py"
        code_path.write_text(code, encoding="utf-8")
        base_path.write_text(base_program, encoding="utf-8")
        process: subprocess.Popen[bytes] | None = None
        started = time.monotonic()
        try:
            process = self._popen(
                [
                    "base",
                    str(code_path),
                    entry_point,
                    "--base-test",
                    str(base_path),
                    "--cpu-limit",
                    str(max(self.cpu_limit_s, 30)),
                    "--fsize-mb",
                    str(self.fsize_mb),
                    "--network",
                    self.network,
                ],
                temp_dir,
                max(self.cpu_limit_s, 30),
            )
            try:
                stdout, _ = process.communicate(timeout=self.base_timeout_s)
            except subprocess.TimeoutExpired:
                self._kill(process)
                return BaseResult("timeout", time.monotonic() - started, 0.0, "base test wall-clock limit", self.backend)
            wall = time.monotonic() - started
            lines = [line for line in stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
            if not lines:
                return BaseResult("crash", wall, 0.0, f"worker exit {process.returncode} with no result", self.backend)
            try:
                result = json.loads(lines[-1])
            except Exception as exc:
                return BaseResult("infra_error", wall, 0.0, f"bad base result: {exc}", self.backend)
            if result.get("ok"):
                return BaseResult("pass", wall, float(result.get("cpu_s", 0.0)), "", self.backend)
            return BaseResult(
                "fail",
                wall,
                float(result.get("cpu_s", 0.0)),
                f"{result.get('error_type', 'Error')}: {result.get('error', '')}"[:500],
                self.backend,
            )
        finally:
            if process is not None and process.poll() is None:
                self._kill(process)
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _evaluate(checker: Any, output: Any, case: dict[str, Any], index: int) -> tuple[bool, str, str]:
        from .testspec import evaluate_checker

        return evaluate_checker(
            checker,
            output,
            case["args"],
            case["expected"],
            int(case.get("source_index", index)),
            float(case.get("atol", 0.0)),
        )
