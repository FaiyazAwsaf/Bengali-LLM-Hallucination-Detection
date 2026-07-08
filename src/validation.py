"""Repeated stratified cross-validation harness.

Ground rules from docs/implementation.md section 5:
- 299 rows is tiny, so every score is reported as mean +/- std across repeated folds,
  never a single point estimate.
- Macro-F1 is always reported three ways: overall, context-present, closed-book.
- Folds are stratified jointly on (label, subgroup) so each fold has proportional
  representation of both classes and both subgroups.
- Every candidate must beat the naive baselines by more than one fold-std, otherwise
  the "improvement" is indistinguishable from noise.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

EXPERIMENTS_LOG = Path(__file__).resolve().parent.parent / "experiments_log.csv"

# fit_predict(train_df, eval_df) -> array of 0/1 predictions for eval_df rows
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def subgroup_scores(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Macro-F1 overall + per subgroup, with per-subgroup confusion matrices."""
    y_true = df["label"].to_numpy()
    y_pred = np.asarray(y_pred).astype(int)
    out = {"overall": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    for name, mask in [("context", ~df["is_closed_book"]), ("closed_book", df["is_closed_book"])]:
        mask = mask.to_numpy()
        out[name] = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        out[f"{name}_confusion"] = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
    return out


@dataclass
class CVResult:
    name: str
    fold_scores: list[dict] = field(default_factory=list)

    def _agg(self, key: str) -> tuple[float, float]:
        vals = [s[key] for s in self.fold_scores]
        return float(np.mean(vals)), float(np.std(vals))

    def summary(self) -> dict:
        out = {"name": self.name}
        for key in ("overall", "context", "closed_book"):
            mean, std = self._agg(key)
            out[f"{key}_mean"] = round(mean, 4)
            out[f"{key}_std"] = round(std, 4)
        return out

    def confusion_totals(self) -> dict:
        """Summed confusion matrices across folds — where label collapse shows up."""
        return {
            key: sum(s[f"{key}_confusion"] for s in self.fold_scores)
            for key in ("context", "closed_book")
        }

    def print_report(self) -> None:
        s = self.summary()
        print(f"\n=== {self.name} ===")
        for key in ("overall", "context", "closed_book"):
            print(f"  {key:12s} Macro-F1: {s[f'{key}_mean']:.3f} +/- {s[f'{key}_std']:.3f}")
        for key, cm in self.confusion_totals().items():
            print(f"  {key} confusion (rows=true 0/1, cols=pred 0/1):\n{cm}")


def run_cv(
    df: pd.DataFrame,
    fit_predict: FitPredict,
    name: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> CVResult:
    """Repeated stratified K-fold on the labeled sample set."""
    strat_key = df["label"].astype(str) + "_" + df["is_closed_book"].astype(str)
    result = CVResult(name=name)
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + repeat)
        for train_idx, val_idx in skf.split(df, strat_key):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            preds = fit_predict(train_df, val_df)
            result.fold_scores.append(subgroup_scores(val_df, preds))
    return result


# --- Naive baselines every model must beat -------------------------------------------------

def baseline_all_ones(train_df, eval_df):
    return np.ones(len(eval_df), dtype=int)


def baseline_all_zeros(train_df, eval_df):
    return np.zeros(len(eval_df), dtype=int)


def baseline_majority_per_subgroup(train_df, eval_df):
    """Predict each subgroup's majority training label."""
    majority = train_df.groupby("is_closed_book")["label"].agg(lambda s: int(s.mode()[0]))
    return eval_df["is_closed_book"].map(majority).fillna(1).astype(int).to_numpy()


NAIVE_BASELINES = {
    "all_ones": baseline_all_ones,
    "all_zeros": baseline_all_zeros,
    "majority_per_subgroup": baseline_majority_per_subgroup,
}


# --- Experiment log (evidence base for the Phase 2 paper) ----------------------------------

def log_experiment(result: CVResult, notes: str = "", runtime_s: float | None = None,
                   weights_gb: float | None = None) -> None:
    row = result.summary()
    row.update(
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        notes=notes,
        runtime_s=runtime_s,
        weights_gb=weights_gb,
    )
    log = pd.DataFrame([row])
    if EXPERIMENTS_LOG.exists():
        log = pd.concat([pd.read_csv(EXPERIMENTS_LOG), log], ignore_index=True)
    log.to_csv(EXPERIMENTS_LOG, index=False)
