from __future__ import annotations

from rewardforge.sandbox import SandboxRunner


def _checker(out, args, expected, index, atol):
    assert out == expected


def _runner() -> SandboxRunner:
    return SandboxRunner(
        {
            "test_timeout_s": 0.5,
            "base_timeout_s": 3.0,
            "load_timeout_s": 5.0,
            "cpu_limit_s": 10,
            "memory_limit_mb": 512,
            "file_size_limit_mb": 2,
            "network": "patch",
            "sandbox_exec": "off",
        }
    )


def _case(value=5, index=0):
    return [{"args": [2, 3], "expected": value, "source_index": index, "atol": 0.0}]


def test_passing_candidate_and_stdout_pollution():
    code = "print('import noise')\ndef add(a, b):\n    print('call noise')\n    return a + b\n"
    result = _runner().run_suite(code, "add", _case(), _checker)
    assert result.status == "ok"
    assert result.all_passed
    assert result.outcomes[0].status == "pass"


def test_wrong_output_and_exception_are_candidate_failures():
    runner = _runner()
    wrong = runner.run_suite("def add(a, b):\n    return a - b\n", "add", _case(), _checker)
    assert wrong.outcomes[0].status == "fail"
    assert wrong.outcomes[0].failure_kind == "wrong_output"

    raises = runner.run_suite("def add(a, b):\n    raise ValueError('nope')\n", "add", _case(), _checker)
    assert raises.outcomes[0].status == "fail"
    assert raises.outcomes[0].failure_kind == "exception:ValueError"


def test_load_error_is_distinguished_from_harness_error():
    result = _runner().run_suite("def add(a, b)\n", "add", _case(), _checker)
    assert result.status == "load_error"
    assert all(outcome.status == "not_run" for outcome in result.outcomes)


def test_timeout_kills_worker():
    code = "def add(a, b):\n    while True:\n        pass\n"
    result = _runner().run_suite(code, "add", _case(), _checker, timeout_s=0.2)
    assert result.status == "timeout"
    assert result.outcomes[0].status == "timeout"


def test_network_is_blocked_best_effort():
    code = (
        "import socket\n"
        "def add(a, b):\n"
        "    socket.create_connection(('example.com', 80), timeout=0.2)\n"
        "    return a + b\n"
    )
    result = _runner().run_suite(code, "add", _case(), _checker)
    assert result.outcomes[0].status == "fail"
    assert result.outcomes[0].failure_kind == "exception:OSError"


def test_base_test_run():
    code = "def add(a, b):\n    return a + b\n"
    result = _runner().run_base(code, "add", "assert add(1, 2) == 3\n")
    assert result.status == "pass"

    bad = _runner().run_base("def add(a, b):\n    return 0\n", "add", "assert add(1, 2) == 3\n")
    assert bad.status == "fail"
