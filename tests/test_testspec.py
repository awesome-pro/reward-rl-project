from __future__ import annotations

from rewardforge.testspec import (
    OVERSIZED,
    evaluate_checker,
    extract_base_call_args,
    extract_features,
    infer_mbpp_entry_point,
    load_checker,
    parse_plus_suite,
)


ASSERTION_SUITE = '''
import math

def assertion(out, exp, atol):
    if out != exp and atol:
        assert abs(out - exp) <= atol
    else:
        assert out == exp

def check(candidate):
    inputs = [[2], [3]]
    results = [4, 9]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assertion(candidate(*inp), exp, 0)
'''

CUSTOM_SUITE = '''
def check(candidate):
    inputs = [[2], [3]]
    for i, inp in enumerate(inputs):
        assert candidate(*inp) == inp[0] ** 2
'''


def test_parse_assertion_suite():
    suite = parse_plus_suite(ASSERTION_SUITE, "square")
    assert suite.inputs == [(2,), (3,)]
    assert suite.expected == [4, 9]
    assert suite.checker_kind == "assertion"
    checker = load_checker(suite.assertion_source)
    assert evaluate_checker(checker, 4, (2,), 4, 0, 0.0)[0]
    assert not evaluate_checker(checker, 5, (2,), 4, 0, 0.0)[0]


def test_parse_custom_check_suite():
    suite = parse_plus_suite(CUSTOM_SUITE, "square")
    assert suite.expected == [None, None]  # the custom body derives it from inp
    assert suite.checker_kind == "custom"
    checker = load_checker(suite.assertion_source)
    assert evaluate_checker(checker, 4, (2,), None, 0, 0.0)[0]
    assert not evaluate_checker(checker, 5, (2,), None, 0, 0.0)[0]


def test_oversized_expected_is_marked():
    code = f'''
def check(candidate):
    inputs = [[1]]
    results = [10 ** {20000}]
    for i, (inp, exp) in enumerate(zip(inputs, results)):
        assert candidate(*inp) == exp
'''
    suite = parse_plus_suite(code, "f")
    assert suite.expected[0] is OVERSIZED


def test_mbpp_entry_point_and_base_args():
    code = "import math\ndef is_even(n):\n    return n % 2 == 0\n"
    tests = ["assert is_even(2) == True", "assert is_even(3) == False"]
    assert infer_mbpp_entry_point(code, tests) == "is_even"
    args = extract_base_call_args("\n".join(tests), {"is_even"})
    assert args == [(2,), (3,)]


def test_features_are_deterministic():
    features = extract_features([[]], [])
    assert features["size_bucket"] == "0-1"
    assert "empty" in features["edge_flags"]
    assert extract_features([[]], []) == features
