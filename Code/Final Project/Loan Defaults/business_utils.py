from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Default financial assumptions; override by defining BUSINESS_PARAMS in the notebook.
DEFAULT_BUSINESS_PARAMS: Dict[str, float] = {
    "ead_method": "limit",          # LIMIT_BAL as the exposure proxy
    "cap_to_limit": True,           # Do not allow EAD to exceed LIMIT_BAL
    "apr": 0.18,                    # Annual percentage rate on approved accounts
    "cost_of_funds": 0.035,         # Cost of capital
    "horizon_months": 12,           # Profit horizon
    "lgd": 0.8,                     # Loss given default (fraction of EAD)
    "origination_cost": 150.0,      # One-time origination expense
    "service_cost_monthly": 2.5,    # Servicing cost per month
    "tn_benefit_flat": 25.0,        # Benefit of correctly rejecting a default
    "collection_cost_flat": 75.0,   # Fixed cost when a default occurs
}

DEFAULT_THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
REQUIRED_ARRAY_COLUMNS = ("B_TP", "C_FP", "B_TN", "C_FN")


def _select_bill_columns(df: pd.DataFrame, stop: int) -> Sequence[str]:
    return [f"BILL_AMT{i}" for i in range(1, stop + 1) if f"BILL_AMT{i}" in df.columns]


def _ensure_dataframe(arrays: pd.DataFrame | Dict[str, Iterable[float]]) -> pd.DataFrame:
    if isinstance(arrays, pd.DataFrame):
        df = arrays.copy()
    else:
        df = pd.DataFrame(arrays)
    missing = [col for col in REQUIRED_ARRAY_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required array columns: {missing}")
    return df


@dataclass
class BusinessModel:
    """Turn BUSINESS_PARAMS into per-loan utility arrays."""

    params: Dict[str, float]

    def __post_init__(self) -> None:
        merged = DEFAULT_BUSINESS_PARAMS.copy()
        merged.update(self.params or {})
        self.params = merged

    # Public API -----------------------------------------------------------------
    def per_loan_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with B_TP, C_FP, B_TN, C_FN, and EAD."""
        ead = self._compute_ead(df)
        p = self.params
        spread = max(p["apr"] - p["cost_of_funds"], 0.0)
        horizon_years = p["horizon_months"] / 12.0

        b_tp = ead * spread * horizon_years
        c_fp = ead * p["lgd"] + p["collection_cost_flat"]
        c_fn = np.maximum(
            0.0,
            b_tp - (p["origination_cost"] + p["service_cost_monthly"] * p["horizon_months"]),
        )
        b_tn = np.full_like(b_tp, p["tn_benefit_flat"], dtype=float)

        arrays = pd.DataFrame(
            {
                "B_TP": b_tp,
                "C_FP": c_fp,
                "B_TN": b_tn,
                "C_FN": c_fn,
                "EAD": ead,
            },
            index=getattr(df, "index", None),
        )
        return arrays

    # Internal helpers ------------------------------------------------------------
    def _compute_ead(self, df: pd.DataFrame) -> np.ndarray:
        method = str(self.params.get("ead_method", "limit")).lower()
        ead: Optional[np.ndarray] = None

        if method == "limit" and "LIMIT_BAL" in df.columns:
            ead = df["LIMIT_BAL"].to_numpy(float, copy=True)
        elif method == "avg_bill3":
            cols = _select_bill_columns(df, 3)
            if cols:
                ead = df[cols].mean(axis=1).to_numpy(float)
        elif method == "max_bill6":
            cols = _select_bill_columns(df, 6)
            if cols:
                ead = df[cols].max(axis=1).to_numpy(float)
        elif method in df.columns:
            ead = df[method].to_numpy(float, copy=True)

        if ead is None:
            fallback_col = next((c for c in ("LIMIT_BAL", "BILL_AMT1") if c in df.columns), None)
            if fallback_col is None:
                raise KeyError("Unable to derive EAD: expected LIMIT_BAL or BILL_AMT columns.")
            ead = df[fallback_col].to_numpy(float, copy=True)

        ead = np.maximum(ead, 0.0)
        if self.params.get("cap_to_limit", False) and "LIMIT_BAL" in df.columns:
            ead = np.minimum(ead, df["LIMIT_BAL"].to_numpy(float))
        return ead


def realized_utility_from_arrays(
    y_true: Sequence[int],
    approve_vector: Sequence[int],
    arrays: pd.DataFrame | Dict[str, Iterable[float]],
    *,
    normalize: bool = True,
) -> float:
    """
    Compute realized (observed) utility given approval decisions built on the arrays.

    Parameters
    ----------
    y_true : iterable of ints (1 = default, 0 = non-default)
    approve_vector : iterable of ints (1 = approve, 0 = reject)
    arrays : DataFrame or dict with B_TP, C_FP, B_TN, C_FN (and optionally EAD)
    normalize : if True, returns per-customer utility; else returns total utility
    """

    y_true = np.asarray(y_true).astype(int)
    approve_vector = np.asarray(approve_vector).astype(int)
    arr_df = _ensure_dataframe(arrays)

    tp_mask = (approve_vector == 1) & (y_true == 0)
    fp_mask = (approve_vector == 1) & (y_true == 1)
    tn_mask = (approve_vector == 0) & (y_true == 1)
    fn_mask = (approve_vector == 0) & (y_true == 0)

    utility = (
        arr_df.loc[tp_mask, "B_TP"].sum()
        - arr_df.loc[fp_mask, "C_FP"].sum()
        + arr_df.loc[tn_mask, "B_TN"].sum()
        - arr_df.loc[fn_mask, "C_FN"].sum()
    )

    if normalize:
        n = max(len(y_true), 1)
        return float(utility) / n
    return float(utility)


def search_best_threshold_arrays(
    y_true: Sequence[int],
    prob_default: Sequence[float],
    arrays: pd.DataFrame | Dict[str, Iterable[float]],
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Evaluate normalized utility for a grid of thresholds and return the best row.
    """

    probs = np.asarray(prob_default, dtype=float)
    arr_df = _ensure_dataframe(arrays)
    if len(arr_df) != len(probs):
        raise ValueError("arrays and prob_default must have the same length.")

    thresh = np.asarray(thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID, dtype=float)
    records = []
    for t in thresh:
        approve = (probs < t).astype(int)
        total_util = realized_utility_from_arrays(y_true, approve, arr_df, normalize=False)
        normalized_util = total_util / len(arr_df)
        records.append(
            {
                "threshold": float(t),
                "normalized_utility": normalized_util,
                "utility": total_util,
                "approval_rate": approve.mean(),
            }
        )

    curve = pd.DataFrame(records)
    best_idx = curve["normalized_utility"].idxmax()
    best_row = curve.loc[best_idx]
    return curve, best_row


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """Return confusion-matrix counts mapped to a dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_utility_scorer(
    biz: BusinessModel,
    *,
    thresholds: Optional[Sequence[float]] = None,
) -> callable:
    """
    Create a GridSearch-compatible scorer that maximizes normalized utility.

    GridSearchCV will call this function with signature scorer(estimator, X, y).
    """

    if biz is None:
        raise ValueError("BusinessModel instance (biz) is required to build the utility scorer.")

    threshold_grid = thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID

    def _score(estimator, X, y) -> float:
        probs = estimator.predict_proba(X)[:, 1]
        if isinstance(X, pd.DataFrame):
            features = X
        else:
            feature_names = getattr(estimator, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError(
                    "Estimator does not expose original feature names; "
                    "pass a pandas DataFrame with column names when fitting."
                )
            features = pd.DataFrame(X, columns=feature_names)
        arrays = biz.per_loan_arrays(features)
        _, best_row = search_best_threshold_arrays(
            y, probs, arrays, thresholds=threshold_grid
        )
        return float(best_row["normalized_utility"])

    _score.__name__ = "utility_scorer"
    return _score


__all__ = [
    "BusinessModel",
    "DEFAULT_BUSINESS_PARAMS",
    "DEFAULT_THRESHOLD_GRID",
    "build_utility_scorer",
    "confusion_counts",
    "realized_utility_from_arrays",
    "search_best_threshold_arrays",
]
