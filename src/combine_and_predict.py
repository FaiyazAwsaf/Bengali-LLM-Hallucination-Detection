"""Build a submission by routing rows to the current best predictor per subgroup.

- context rows  -> Track A zero-shot NLI, threshold calibrated on all 130 labeled context rows
- closed-book rows -> Track B classifier (agreement features from independent TigerLLM
  answers) when its feature caches exist for both splits; otherwise majority class (0).

Always validates the file with submission_check before it is considered submittable.

Usage: python src/combine_and_predict.py <output_name.csv>
"""

import sys
from pathlib import Path

import pandas as pd

from data_loading import load_samples, load_test
from submission_check import check_submission
from track_a_nli import MAJORITY_CLOSED_BOOK, TEST_SCORES, TRAIN_SCORES, best_threshold, scores_for

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


def predict_context(train: pd.DataFrame, test: pd.DataFrame, sub: pd.DataFrame) -> None:
    train_scores = scores_for(train, TRAIN_SCORES)
    test_scores = scores_for(test, TEST_SCORES)  # requires cache (run track_a_nli.py test)
    ctx = train[~train["is_closed_book"]]
    threshold = best_threshold(
        train_scores.loc[ctx.index, "nli_score"].to_numpy(), ctx["label"].to_numpy()
    )
    print(f"track A threshold (fit on all {len(ctx)} labeled context rows): {threshold:.3f}")
    ctx_mask = ~test["is_closed_book"]
    sub.loc[ctx_mask, "label"] = (test_scores.loc[ctx_mask, "nli_score"] >= threshold).astype(int)


def predict_closed_book(train: pd.DataFrame, test: pd.DataFrame, sub: pd.DataFrame) -> None:
    from track_b_consistency import FEATURE_COLUMNS, features_cache_path

    cb_mask = test["is_closed_book"]
    train_feats_path = features_cache_path("train")
    test_feats_path = features_cache_path("test")
    if not (train_feats_path.exists() and test_feats_path.exists()):
        print("track B feature caches missing -> closed-book falls back to majority class "
              f"({MAJORITY_CLOSED_BOOK})")
        sub.loc[cb_mask, "label"] = MAJORITY_CLOSED_BOOK
        return

    from track_b_classifier import make_closed_book_model

    train_feats = pd.read_csv(train_feats_path, index_col=0)
    test_feats = pd.read_csv(test_feats_path, index_col=0)
    cb_train = train[train["is_closed_book"]]
    model = make_closed_book_model()
    model.fit(train_feats.loc[cb_train.index, FEATURE_COLUMNS], cb_train["label"])
    sub.loc[cb_mask, "label"] = model.predict(test_feats.loc[cb_mask[cb_mask].index, FEATURE_COLUMNS])
    print(f"track B classifier predicted {int(cb_mask.sum())} closed-book rows")


def build_submission(out_name: str) -> Path:
    train = load_samples()
    test = load_test()

    sub = test[["id"]].copy()
    sub["label"] = MAJORITY_CLOSED_BOOK
    predict_context(train, test, sub)
    predict_closed_book(train, test, sub)
    sub["label"] = sub["label"].astype(int)

    out_path = SUBMISSIONS_DIR / out_name
    sub.to_csv(out_path, index=False)

    problems = check_submission(out_path)
    if problems:
        for p in problems:
            print(f"  INVALID: {p}")
        sys.exit(1)
    ctx_mask = ~test["is_closed_book"]
    print(f"OK: {out_path} ({len(sub)} rows, mean label {sub['label'].mean():.3f}, "
          f"context mean {sub.loc[ctx_mask, 'label'].mean():.3f}, "
          f"closed-book mean {sub.loc[~ctx_mask, 'label'].mean():.3f})")
    return out_path


if __name__ == "__main__":
    build_submission(sys.argv[1] if len(sys.argv) > 1 else "combined.csv")
