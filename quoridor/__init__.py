"""AlphaZero-style Quoridor engine.

``pyrules`` is the readable reference implementation of the rules; ``fastrules``
is the Numba-jitted core used by search. They are kept in exact agreement by the
differential tests in ``tests/test_rules.py``.
"""

from quoridor import fastrules, pyrules

__all__ = ["fastrules", "pyrules"]
__version__ = "0.1.0"
