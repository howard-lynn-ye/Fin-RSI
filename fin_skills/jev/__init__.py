"""Financial JEV question templates and validation (#30), plus v2's retained local models.

Templates and the validator make no implicit model or network calls. AnalyticalStockJEV and the
neural adapter are retained from v2 pending a separate decision (see fin_skills/jev/stock_jev.py).
"""
from fin_skills.jev.stock_jev import (
    ORDINAL_RATING_LEVELS,
    REGIME_CHOICE_OPTIONS,
    AnalyticalStockJEV,
    audit_stock_jev_causality,
    build_disqualification_question,
    build_macro_regime_question,
    build_ordinal_rating_question,
    validate_stock_jev_decision,
)


def __getattr__(name):
    # The neural adapter imports torch; load it only on use, so template imports stay torch-free.
    if name in ("NeuralStockJEVAdapter", "NeuralStockJEVNet"):
        from fin_skills.jev import neural_jev
        return getattr(neural_jev, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ORDINAL_RATING_LEVELS", "REGIME_CHOICE_OPTIONS", "AnalyticalStockJEV",
    "NeuralStockJEVAdapter", "NeuralStockJEVNet", "audit_stock_jev_causality",
    "build_disqualification_question", "build_macro_regime_question",
    "build_ordinal_rating_question", "validate_stock_jev_decision",
]
