"""Build a submission by routing rows to the current best predictor per subgroup.

Current state (Milestone 1):
- context rows  -> Track A zero-shot NLI, threshold calibrated on all 130 labeled context rows
- closed-book rows -> majority class (0) until Track B exists

Always validates the file with submission_check before it is considered submittable.

Usage: python src/combine_and_predict.py <output_name.csv>
"""

import sys
from pathlib import Path

from data_loading import load_samples, load_test
from submission_check import check_submission
from track_a_nli import MAJORITY_CLOSED_BOOK, TEST_SCORES, TRAIN_SCORES, best_threshold, scores_for

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"


def build_submission(out_name: str) -> Path:
    train = load_samples()
    test = load_test()

    train_scores = scores_for(train, TRAIN_SCORES)
    test_scores = scores_for(test, TEST_SCORES)  # requires the cache (run track_a_nli.py test)

    ctx = train[~train["is_closed_book"]]
    threshold = best_threshold(
        train_scores.loc[ctx.index, "nli_score"].to_numpy(), ctx["label"].to_numpy()
    )
    print(f"threshold (fit on all {len(ctx)} labeled context rows): {threshold:.3f}")

    sub = test[["id"]].copy()
    sub["label"] = MAJORITY_CLOSED_BOOK
    ctx_mask = ~test["is_closed_book"]
    sub.loc[ctx_mask, "label"] = (
        test_scores.loc[ctx_mask, "nli_score"] >= threshold
    ).astype(int)

    out_path = SUBMISSIONS_DIR / out_name
    sub.to_csv(out_path, index=False)

    problems = check_submission(out_path)
    if problems:
        for p in problems:
            print(f"  INVALID: {p}")
        sys.exit(1)
    print(f"OK: {out_path} ({len(sub)} rows, mean label {sub['label'].mean():.3f}, "
          f"context mean {sub.loc[ctx_mask, 'label'].mean():.3f})")
    return out_path


if __name__ == "__main__":
    build_submission(sys.argv[1] if len(sys.argv) > 1 else "nli_ctx_majority_cb.csv")
