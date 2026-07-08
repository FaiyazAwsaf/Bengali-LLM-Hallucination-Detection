"""Track B classifier + combined two-track CV.

Small logistic regression (169 labeled closed-book rows — nothing deeper is
justified, implementation.md 3.4) on the agreement features from
track_b_consistency.py. Evaluated only through the repeated-CV harness, combined
with Track A so the overall number is directly comparable to the leaderboard.

Usage: python src/track_b_classifier.py   (needs the train feature cache)
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loading import load_samples
from track_a_nli import TRAIN_SCORES, best_threshold, scores_for
from track_b_consistency import FEATURE_COLUMNS, features_cache_path
from validation import log_experiment, run_cv


def make_closed_book_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def make_fit_predict(nli_scores: pd.DataFrame, features: pd.DataFrame):
    """Two-track fit_predict for the harness: Track A on context, Track B on closed-book."""

    def fit_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(eval_df), dtype=int)

        # Track A: threshold calibrated on the train fold's context rows
        train_ctx = train_df[~train_df["is_closed_book"]]
        t = best_threshold(
            nli_scores.loc[train_ctx.index, "nli_score"].to_numpy(),
            train_ctx["label"].to_numpy(),
        )
        ctx_mask = (~eval_df["is_closed_book"]).to_numpy()
        preds[ctx_mask] = (
            nli_scores.loc[eval_df.index[ctx_mask], "nli_score"].to_numpy() >= t
        ).astype(int)

        # Track B: logistic regression trained on the train fold's closed-book rows
        train_cb = train_df[train_df["is_closed_book"]]
        model = make_closed_book_model()
        model.fit(features.loc[train_cb.index, FEATURE_COLUMNS], train_cb["label"])
        cb_mask = eval_df["is_closed_book"].to_numpy()
        preds[cb_mask] = model.predict(features.loc[eval_df.index[cb_mask], FEATURE_COLUMNS])
        return preds

    return fit_predict


def main() -> None:
    df = load_samples()
    nli_scores = scores_for(df, TRAIN_SCORES)
    features = pd.read_csv(features_cache_path("train"), index_col=0)

    result = run_cv(df, make_fit_predict(nli_scores, features), name="nli_ctx+trackB_cb")
    result.print_report()
    log_experiment(result, notes="track B consistency features (TigerLLM regen + LaBSE + NLI agreement)")

    # which features carry weight (fit on all closed-book rows, standardized coefs)
    cb = df[df["is_closed_book"]]
    model = make_closed_book_model().fit(features.loc[cb.index, FEATURE_COLUMNS], cb["label"])
    coefs = model.named_steps["logisticregression"].coef_[0]
    print("\nclosed-book feature coefficients (standardized):")
    for name, c in sorted(zip(FEATURE_COLUMNS, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name:14s} {c:+.3f}")


if __name__ == "__main__":
    main()
