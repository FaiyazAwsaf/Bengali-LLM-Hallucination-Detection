"""Load the competition data and split rows into the two subgroups.

Subgroups (see docs/implementation.md section 1):
- context-present rows -> entailment problem (Track A)
- closed-book rows (context == "[NULL]") -> factual-verification problem (Track B)
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SAMPLES_PATH = DATA_DIR / "dataset samples.json"
TEST_PATH = DATA_DIR / "test set.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample submission.csv"

NULL_CONTEXT = "[NULL]"


def _add_subgroup_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["context"] = df["context"].fillna(NULL_CONTEXT).astype(str).str.strip()
    df["is_closed_book"] = df["context"] == NULL_CONTEXT
    return df


def load_samples() -> pd.DataFrame:
    """299 labeled rows: context, prompt_bn, response_bn, label, is_closed_book."""
    df = pd.read_json(SAMPLES_PATH)
    df = _add_subgroup_flag(df)
    df["label"] = df["label"].astype(int)
    return df


def load_test() -> pd.DataFrame:
    """2,516 unlabeled rows: id, context, prompt_bn, response_bn, is_closed_book."""
    df = pd.read_csv(TEST_PATH)
    return _add_subgroup_flag(df)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_SUBMISSION_PATH)


if __name__ == "__main__":
    train = load_samples()
    test = load_test()
    for name, df in [("samples", train), ("test", test)]:
        n_closed = int(df["is_closed_book"].sum())
        print(f"{name}: {len(df)} rows | context-present {len(df) - n_closed} | closed-book {n_closed}")
    print("\nlabel balance (samples):")
    print(train.groupby("is_closed_book")["label"].agg(["count", "mean"]))
