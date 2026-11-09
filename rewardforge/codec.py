"""A JSON-safe value codec for test arguments and candidate return values.

The benchmark inputs contain tuples, sets, floats such as ``inf`` and (in a
few tasks) numpy scalars.  We need to preserve those types across the
supervisor/worker process boundary without ever unpickling data produced by
the candidate.  Pickle would execute candidate-controlled code in the trusted
process, so the protocol uses this explicit, tagged representation instead.
"""

from __future__ import annotations

import base64
import math
from typing import Any


class OpaqueValue:
    """Sentinel for a value that could not be represented safely."""

    def __init__(self, type_name: str, text: str) -> None:
        self.type_name = type_name
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"OpaqueValue({self.type_name!r}, {self.text!r})"

    def __eq__(self, other: object) -> bool:
        return False


def encode_value(value: Any) -> Any:
    # Primitive JSON values.  bool must be checked before int.
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
        return {"__rf__": "set", "v": _stable_sort([encode_value(item) for item in value])}
    if isinstance(value, frozenset):
        return {"__rf__": "frozenset", "v": _stable_sort([encode_value(item) for item in value])}
    if isinstance(value, dict):
        return {
            "__rf__": "dict",
            "v": [[encode_value(key), encode_value(item)] for key, item in value.items()],
        }
    # numpy is optional for the harness itself, but candidates and benchmark
    # assertions may use it.  Import lazily so the core stays stdlib-only.
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy is a declared dependency
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return {
                "__rf__": "ndarray",
                "dtype": str(value.dtype),
                "v": [encode_value(item) for item in value.tolist()],
            }
        if isinstance(value, np.generic):
            return encode_value(value.item())
    return {
        "__rf__": "opaque",
        "type": type(value).__qualname__,
        "repr": repr(value)[:500],
    }


def decode_value(value: Any) -> Any:
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
    if tag == "ndarray":
        try:
            import numpy as np  # type: ignore

            return np.asarray([decode_value(item) for item in value["v"]], dtype=value["dtype"])
        except Exception:
            return [decode_value(item) for item in value["v"]]
    if tag == "opaque":
        return OpaqueValue(value.get("type", "unknown"), value.get("repr", ""))
    raise ValueError(f"Unknown codec tag: {tag!r}")


def _stable_sort(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda item: repr(item))
