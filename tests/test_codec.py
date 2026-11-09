from __future__ import annotations

import math

from rewardforge.codec import OpaqueValue, decode_value, encode_value


def test_roundtrip_nested_values():
    value = {"a": (1, 2.5, [True, None]), "b": {"x", "y"}, "c": b"\x00\xff"}
    decoded = decode_value(encode_value(value))
    assert decoded["a"] == (1, 2.5, [True, None])
    assert decoded["b"] == {"x", "y"}
    assert decoded["c"] == b"\x00\xff"


def test_float_special_values():
    assert math.isnan(decode_value(encode_value(float("nan"))))
    assert decode_value(encode_value(float("inf"))) == float("inf")
    assert decode_value(encode_value(float("-inf"))) == float("-inf")


def test_opaque_value_does_not_raise():
    decoded = decode_value(encode_value(object()))
    assert isinstance(decoded, OpaqueValue)
    assert decoded != object()
