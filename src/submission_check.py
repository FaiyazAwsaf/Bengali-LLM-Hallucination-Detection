"""Validate a submission CSV before uploading.

The rules state that missing rows, missing labels, non-integer values, or extra
columns cause outright rejection with NO score — and we only get 4 submissions/day.
Run this on every candidate file before `kaggle competitions submit`.

Usage: python src/submission_check.py submissions/my_submission.csv
"""

import sys
from pathlib import Path

import pandas as pd

from data_loading import load_test


def check_submission(path: str | Path) -> list[str]:
    """Return a list of problems; empty list means the file is safe to submit."""
    problems = []
    path = Path(path)
    if not path.exists():
        return [f"file not found: {path}"]

    df = pd.read_csv(path)

    if list(df.columns) != ["id", "label"]:
        problems.append(f"columns must be exactly ['id', 'label'], got {list(df.columns)}")
        return problems

    if df["id"].isna().any() or df["label"].isna().any():
        problems.append("found missing values in id or label")

    if not pd.api.types.is_integer_dtype(df["label"]):
        problems.append(f"label dtype must be integer, got {df['label'].dtype}")
    elif not df["label"].isin([0, 1]).all():
        bad = sorted(df.loc[~df["label"].isin([0, 1]), "label"].unique().tolist())
        problems.append(f"labels must be 0 or 1, found {bad}")

    expected_ids = set(load_test()["id"])
    got_ids = set(df["id"])
    if df["id"].duplicated().any():
        problems.append(f"{df['id'].duplicated().sum()} duplicate ids")
    if got_ids != expected_ids:
        missing, extra = expected_ids - got_ids, got_ids - expected_ids
        if missing:
            problems.append(f"{len(missing)} test ids missing (e.g. {sorted(missing)[:5]})")
        if extra:
            problems.append(f"{len(extra)} unknown ids (e.g. {sorted(extra)[:5]})")

    return problems


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    problems = check_submission(sys.argv[1])
    if problems:
        print(f"REJECTED — do not submit {sys.argv[1]}:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    print(f"OK to submit: {sys.argv[1]} ({len(df)} rows, mean label {df['label'].mean():.3f})")


if __name__ == "__main__":
    main()
