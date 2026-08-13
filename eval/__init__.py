"""SDK evaluate() — offline experiments from code."""

from __future__ import annotations

from traccia.eval.builtins import BUILTIN_SCORERS, run_builtin_scorer
from traccia.eval.evaluate import EvaluateResult, evaluate
from traccia.eval.errors import EvaluateError

__all__ = [
    "evaluate",
    "EvaluateResult",
    "EvaluateError",
    "BUILTIN_SCORERS",
    "run_builtin_scorer",
]
