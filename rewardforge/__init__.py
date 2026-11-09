"""RewardForge: cost-controlled executable reward verification.

The first experiment is a CPU-only audit and offline replay study.  It does
not train a model.  See README.md for the experiment contract.
"""

__version__ = "0.1.0"

# Benchmark values include very large integers; Python's default decimal
# conversion limit is irrelevant here and breaks JSON serialization.
import sys as _sys

if hasattr(_sys, "set_int_max_str_digits"):
    _sys.set_int_max_str_digits(0)
