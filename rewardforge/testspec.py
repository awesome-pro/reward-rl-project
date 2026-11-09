"""Parsing and validation of EvalPlus test suites.

EvalPlus stores tests as Python source.  HumanEval+ puts
``inputs``/``results`` (or a ``ref_func``) inside ``check``; MBPP+ puts them
at module level and loops over them.  We parse both forms, evaluate the
trusted reference on each input once during data preparation, and store one
self-contained case per test.  The candidate process only ever receives the
arguments for the current case: it never sees expected values, the assertion,
or any audit test.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import math
import signal
import textwrap
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from .codec import decode_value, encode_value
from .config import stable_key


class TestParseError(ValueError):
    pass


class OversizedValue:
    """Marker for a test whose expected value is too large to be useful."""

    def __repr__(self) -> str:
        return "<oversized>"


OVERSIZED = OversizedValue()
MAX_EXPECTED_CHARS = 16384


class TimeoutError_(TimeoutError):
    pass


@dataclass
class PlusSuite:
    """A parsed, still-unvalidated EvalPlus test suite."""

    inputs: list[tuple[Any, ...]]
    expected: list[Any]
    atols: list[float]
    assertion_source: str
    source_kind: str
    n_source_cases: int
    checker_kind: str = "assertion"


@dataclass
class TestCase:
    index: int
    args: list[Any]
    expected: Any
    atol: float
    features: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        return stable_key(encode_value(tuple(self.args)), encode_value(self.expected))


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Main-thread wall-clock timeout used only for trusted reference code."""

    if seconds <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError_("trusted reference exceeded time limit")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _find_assignment(tree: ast.Module, name: str) -> ast.Assign | None:
    """Prefer a module-level assignment, then a nested one (inside check)."""

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    return None


def _is_bare_call(node: ast.stmt, names: set[str]) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return isinstance(func, ast.Name) and func.id in names


def _filtered_module_source(tree: ast.Module, source: str, skip: set[str]) -> str:
    """Keep imports, functions, and small assignments; drop test loops.

    This is only ever executed for trusted benchmark test code.  The purpose
    is to expose ``assertion``/``ref_func`` without running the test loop.
    """

    segments: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.For, ast.While)):
            continue
        if _is_bare_call(node, {"check", "test", "main", "run_tests"}):
            continue
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & skip:
                continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in skip:
            continue
        segment = ast.get_source_segment(source, node)
        if segment:
            segments.append(segment)
    return "\n\n".join(segments) + "\n"


def _safe_exec(source: str, filename: str = "<test-harness>") -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "rewardforge_test_harness"}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - trusted benchmark data
    return namespace


def _literal_or_eval(node: ast.AST, namespace: dict[str, Any]) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        expression = ast.Expression(body=node)  # type: ignore[arg-type]
        ast.fix_missing_locations(expression)
        return eval(compile(expression, "<test-value>", "eval"), namespace)  # noqa: S307


def _extract_atol(tree: ast.Module, namespace: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "assertion"):
            continue
        if len(node.args) < 3:
            values.append(0.0)
            continue
        try:
            value = _literal_or_eval(node.args[2], namespace)
            values.append(float(value))
        except Exception:
            values.append(0.0)
    return values


def _normalise_input(raw: Any) -> tuple[Any, ...]:
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    # A scalar test input.  EvalPlus normally writes [arg1, arg2], but a few
    # generated suites use a bare value; wrapping it is the deterministic
    # interpretation for a one-argument entry point.
    return (raw,)


def _loop_calls_candidate(loop: ast.For, entry_point: str) -> bool:
    names = {"candidate", entry_point} - {""}
    for node in ast.walk(loop):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            return True
    return False


def _find_test_loop(tree: ast.Module) -> ast.For | None:
    candidates: list[ast.For] = []
    for node in tree.body:
        if isinstance(node, ast.For):
            candidates.append(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and node not in candidates:
            # Only loops inside check() are relevant; helpers can contain loops.
            candidates.append(node)
    for loop in candidates:
        if _loop_calls_candidate(loop, ""):
            return loop
    # The entry point name is not known here; accept a loop whose body asserts
    # on a call and whose iterable enumerates inputs.
    for loop in candidates:
        text = ast.unparse(loop.iter)
        if "inputs" in text:
            return loop
    return None


class _CallToOutput(ast.NodeTransformer):
    def __init__(self, entry_point: str) -> None:
        self.names = {"candidate", entry_point} - {""}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.names:
            return ast.copy_location(ast.Name(id="out", ctx=ast.Load()), node)
        if isinstance(node.func, ast.Name) and node.func.id == "ref_func":
            return ast.copy_location(ast.Name(id="exp", ctx=ast.Load()), node)
        return node


def _build_custom_check_body(loop: ast.For, entry_point: str) -> str:
    body = [copy.deepcopy(statement) for statement in loop.body]
    transformer = _CallToOutput(entry_point)
    transformed = [transformer.visit(statement) for statement in body]
    module = ast.Module(body=transformed, type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        source = ast.unparse(module)
        compile(source, "<custom-check>", "exec")
    except Exception as exc:
        raise TestParseError(f"custom test body cannot be compiled: {exc!r}") from exc
    # ast.unparse emits the statements at column zero; indentation is added by
    # the caller when embedding them into __rf_check__.
    return source.strip("\n")


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------


def parse_plus_suite(test_code: str, entry_point: str, *, ref_timeout_s: float = 1.5) -> PlusSuite:
    """Parse a HumanEval+/MBPP+ ``test`` source string.

    ``expected`` is materialised here so the audit supervisor never needs to
    re-run a reference implementation during grading.
    """

    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        raise TestParseError(f"test source is not valid Python: {exc}") from exc

    # Evaluate the module without its test loop.  This gives us imports,
    # helpers, and (for MBPP+) module-level inputs/results.
    filtered = _filtered_module_source(tree, test_code, skip={"inputs", "results"})
    try:
        namespace = _safe_exec(filtered)
    except Exception as exc:
        raise TestParseError(f"test prelude failed to execute: {exc!r}") from exc

    # Recover inputs/results even when they were inside check(): execute the
    # assignment expressions in the already-prepared namespace.
    inputs = namespace.get("inputs")
    results = namespace.get("results")
    if inputs is None:
        assignment = _find_assignment(tree, "inputs")
        if assignment is None:
            raise TestParseError("no inputs assignment found")
        inputs = _literal_or_eval(assignment.value, namespace)
    if results is None:
        assignment = _find_assignment(tree, "results")
        if assignment is not None:
            results = _literal_or_eval(assignment.value, namespace)

    if not isinstance(inputs, (list, tuple)):
        raise TestParseError(f"inputs must be a sequence, got {type(inputs)!r}")

    ref_func = namespace.get("ref_func")
    if results is not None and len(results) != len(inputs):
        raise TestParseError(f"inputs/results length mismatch: {len(inputs)} vs {len(results)}")

    expected: list[Any] = []
    for index, raw_input in enumerate(inputs):
        if results is not None:
            value = results[index]
        elif callable(ref_func):
            args = _normalise_input(raw_input)
            try:
                with time_limit(ref_timeout_s):
                    value = ref_func(*args)  # type: ignore[misc]
            except Exception as exc:
                raise TestParseError(f"ref_func failed on case {index}: {exc!r}") from exc
        else:
            # A custom assertion may compute the reference from the input
            # itself (e.g. ``assert candidate(*inp) == expected_fn(*inp)``).
            # Validation against the canonical solution will quarantine it if
            # it actually needs a materialised expected value.
            expected.append(None)
            continue
        try:
            too_large = len(repr(value)) > MAX_EXPECTED_CHARS
        except Exception:
            too_large = True
        expected.append(OVERSIZED if too_large else value)

    atols = _extract_atol(tree, namespace)
    if not atols:
        atols = [0.0] * len(inputs)
    elif len(atols) == 1:
        atols = atols * len(inputs)
    elif len(atols) != len(inputs):
        # Multiple assertion calls with a different grouping: be conservative
        # and use the first value for every case.  The suite is still checked
        # against the canonical solution before it can enter the bank.
        atols = [atols[0]] * len(inputs)

    # Keep imports/helpers needed to re-create the checker at grade time.
    harness_nodes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            segment = ast.get_source_segment(test_code, node)
            if segment:
                harness_nodes.append(segment)
        elif isinstance(node, ast.FunctionDef) and node.name not in {"check", "ref_func"}:
            segment = ast.get_source_segment(test_code, node)
            if segment:
                harness_nodes.append(segment)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names and not (names & {"inputs", "results"}):
                segment = ast.get_source_segment(test_code, node)
                if segment:
                    harness_nodes.append(segment)

    harness_source = "\n\n".join(harness_nodes) + "\n"
    try:
        harness_ns = _safe_exec(harness_source)
    except Exception as exc:
        raise TestParseError(f"test harness failed to execute: {exc!r}") from exc

    checker_kind = "assertion"
    if callable(harness_ns.get("assertion")):
        if results is None and not callable(ref_func):
            raise TestParseError("assertion suite has neither results nor ref_func")
        harness_source += (
            "\n\ndef __rf_check__(out, inp, exp, i, atol):\n"
            "    assertion(out, exp, atol)\n"
        )
    else:
        loop = _find_test_loop(tree)
        if loop is None:
            raise TestParseError("test source defines neither assertion() nor a test loop")
        body = _build_custom_check_body(loop, entry_point)
        harness_source += (
            "\n\ndef __rf_check__(out, inp, exp, i, atol):\n"
            + textwrap.indent(body, "    ")
            + "\n"
        )
        checker_kind = "custom"

    try:
        _safe_exec(harness_source)
    except Exception as exc:
        raise TestParseError(f"test checker failed to compile: {exc!r}") from exc

    return PlusSuite(
        inputs=[_normalise_input(raw) for raw in inputs],
        expected=expected,
        atols=atols,
        assertion_source=harness_source,
        source_kind="top_level" if isinstance(tree.body[0], (ast.Import, ast.ImportFrom)) else "check",
        n_source_cases=len(inputs),
        checker_kind=checker_kind,
    )


def load_checker(assertion_source: str):
    namespace = _safe_exec(assertion_source)
    checker = namespace.get("__rf_check__")
    if not callable(checker):
        raise TestParseError("stored test checker is not callable")
    return checker


def evaluate_checker(
    checker: Any,
    output: Any,
    args: tuple[Any, ...] | list[Any],
    expected: Any,
    index: int,
    atol: float,
) -> tuple[bool, str, str]:
    """Return (passed, failure_kind, detail).  Never raises for a wrong output."""

    try:
        checker(output, args, copy.deepcopy(expected), index, atol)
        return True, "", ""
    except AssertionError as exc:
        return False, "wrong_output", str(exc)[:500]
    except Exception as exc:  # a malformed return value can break the checker
        return False, "comparison_error", f"{type(exc).__name__}: {exc}"[:500]


# ---------------------------------------------------------------------------
# Base tests
# ---------------------------------------------------------------------------


def infer_mbpp_entry_point(code: str, test_sources: Iterable[str]) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    defs = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    calls: list[str] = []
    for source in test_sources:
        try:
            test_tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    for name in calls:
        if name in defs:
            return name
    return defs[0] if defs else ""


def extract_base_call_args(base_code: str, function_names: Iterable[str]) -> list[tuple[Any, ...]]:
    """Literal argument tuples passed to the candidate in base tests.

    Used only to avoid re-counting a base case as an "additional" test.  Calls
    with variables/loops (e.g. HumanEval/38) are intentionally skipped.
    """

    names = set(function_names)
    try:
        tree = ast.parse(base_code)
    except SyntaxError:
        return []
    found: list[tuple[Any, ...]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id in names):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        if any(keyword.arg is None for keyword in node.keywords):
            continue
        try:
            args = tuple(ast.literal_eval(arg) for arg in node.args)
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except Exception:
            continue
        if kwargs:
            # The plus suites in this benchmark are positional.  Store a
            # stable marker rather than pretending positional and keyword
            # calls are duplicates.
            found.append(args + (("__kwargs__", tuple(sorted(kwargs.items()))),))
        else:
            found.append(args)
    return found


def base_test_program(benchmark: str, base_test_code: str, entry_point: str, setup_code: str = "") -> str:
    """Build a self-contained script that raises AssertionError on failure."""

    if benchmark == "humaneval":
        body = base_test_code
        if "def check(" in body:
            return body + f"\n\ncheck({entry_point})\n"
        return body
    # MBPP: original asserts run at module scope.  Execute setup first.
    return (setup_code or "") + "\n" + base_test_code + "\n"


# ---------------------------------------------------------------------------
# Canonical validation
# ---------------------------------------------------------------------------


def build_canonical_code(benchmark: str, prompt: str, canonical_solution: str) -> str:
    if benchmark == "humaneval":
        return prompt + canonical_solution + "\n"
    return canonical_solution + "\n"


def execute_candidate_code(code: str, entry_point: str, timeout_s: float = 5.0) -> tuple[Any, dict[str, Any]]:
    """Load trusted code in-process and return (function, module namespace)."""

    namespace: dict[str, Any] = {"__name__": "rewardforge_canonical"}
    with time_limit(timeout_s):
        exec(compile(code, "<canonical>", "exec"), namespace)  # noqa: S102 - trusted reference
    function = namespace.get(entry_point)
    if not callable(function):
        raise TestParseError(f"entry point {entry_point!r} is not defined")
    return function, namespace


def validate_suite_on_canonical(
    code: str,
    entry_point: str,
    suite: PlusSuite,
    *,
    timeout_s: float = 1.5,
    repeats: int = 1,
) -> tuple[list[bool], list[str]]:
    """Return per-case validity flags and reasons, using the canonical code."""

    function, _ = execute_candidate_code(code, entry_point, timeout_s=timeout_s * 4)
    checker = load_checker(suite.assertion_source)
    valid: list[bool] = []
    reasons: list[str] = []
    for index, args in enumerate(suite.inputs):
        outcomes: list[bool] = []
        reason = ""
        for _ in range(max(1, repeats)):
            try:
                with time_limit(timeout_s):
                    output = function(*args)
                passed, kind, detail = evaluate_checker(
                    checker, output, args, suite.expected[index], index, suite.atols[index]
                )
                outcomes.append(passed)
                if not passed:
                    reason = f"{kind}: {detail}"
            except TimeoutError_:
                outcomes.append(False)
                reason = "timeout"
            except Exception as exc:
                outcomes.append(False)
                reason = f"canonical_error: {type(exc).__name__}: {exc}"
        valid.append(all(outcomes))
        reasons.append(reason)
    return valid, reasons


# ---------------------------------------------------------------------------
# Test features
# ---------------------------------------------------------------------------


def type_signature(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, list):
        return "list[" + "|".join(sorted({type_signature(item) for item in value}))[:60] + "]"
    if isinstance(value, tuple):
        return "tuple[" + "|".join(sorted({type_signature(item) for item in value}))[:60] + "]"
    if isinstance(value, (set, frozenset)):
        return "set[" + "|".join(sorted({type_signature(item) for item in value}))[:60] + "]"
    if isinstance(value, dict):
        return "dict"
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return f"ndarray[{value.dtype}]"
        if isinstance(value, np.generic):
            return type_signature(value.item())
    except Exception:
        pass
    return type(value).__name__


def _size_of(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 1 if abs(value) <= 1 else min(20, 1 + int(math.log10(abs(value))))
    return 1


def _edge_flags(args: list[Any], expected: Any) -> tuple[str, ...]:
    flags: set[str] = set()

    def visit(value: Any) -> None:
        if value is None:
            flags.add("none")
        elif isinstance(value, bool):
            flags.add("bool")
        elif isinstance(value, int):
            if value == 0:
                flags.add("zero")
            if value < 0:
                flags.add("negative")
            if abs(value) >= 10**6:
                flags.add("large")
        elif isinstance(value, float):
            flags.add("float")
            if math.isinf(value) or math.isnan(value):
                flags.add("nonfinite")
        elif isinstance(value, str):
            if value == "":
                flags.add("empty")
            if value.strip() != value:
                flags.add("whitespace")
        elif isinstance(value, (list, tuple, set, frozenset)):
            if len(value) == 0:
                flags.add("empty")
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if not value:
                flags.add("empty")
            for key, item in value.items():
                visit(key)
                visit(item)

    for arg in args:
        visit(arg)
    visit(expected)
    return tuple(sorted(flags))


def extract_features(args: list[Any], expected: Any) -> dict[str, Any]:
    arg_sizes = [_size_of(arg) for arg in args]
    total_size = sum(arg_sizes)
    if total_size <= 1:
        size_bucket = "0-1"
    elif total_size <= 5:
        size_bucket = "2-5"
    elif total_size <= 20:
        size_bucket = "6-20"
    elif total_size <= 100:
        size_bucket = "21-100"
    else:
        size_bucket = "100+"
    arg_types = tuple(type_signature(arg) for arg in args)
    flags = _edge_flags(args, expected)
    expected_type = type_signature(expected)
    expected_size = _size_of(expected)
    cluster = "|".join(
        [
            f"arity={len(args)}",
            "types=" + ",".join(arg_types),
            f"size={size_bucket}",
            "flags=" + ",".join(flags),
            f"ret={expected_type}",
        ]
    )
    return {
        "arity": len(args),
        "arg_types": list(arg_types),
        "arg_sizes": arg_sizes,
        "total_size": total_size,
        "size_bucket": size_bucket,
        "edge_flags": list(flags),
        "expected_type": expected_type,
        "expected_size": expected_size,
        "cluster": cluster,
    }
