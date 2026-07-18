"""Build the Lane 2 (context cross-encoder) fine-tuning set.

Combines two external sources into (context, prompt_bn, response_bn, label, source_id,
origin) rows matching our competition schema:

- BenHalluEval ground-truth QA (data/banglahallueval_qa_1000.csv) -> label 1 rows.
- BenHalluEval generated hallucinations (data/external/BanglaHalluEval/
  "Hallucination Generated Answers"/qa_4000.csv) -> label 0 rows, SAME context/question
  as their source_id's ground-truth row. This same-context contrastive pairing is
  deliberate: it stops a fine-tuned model from learning "which dataset does this look
  like" instead of "does this response match this context" (see docs/session discussion
  on trainset_context.csv's single-source-per-label shortcut risk).
- data/trainset_context.csv's 104 sample-real rows -> both labels, drawn straight from
  our real competition sample (gold, small, always kept).

Every row carries a `source_id` used for a GROUP-AWARE split: two answer-variants of the
same underlying passage must never land on opposite sides of train/val, or the model
partially memorizes the passage during training and validation looks better than it is.

Usage: python src/lane2_dataset.py
Outputs: data/cache/lane2_train.csv, data/cache/lane2_val.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

GT_PATH = DATA_DIR / "banglahallueval_qa_1000.csv"
GEN_PATH = DATA_DIR / "external" / "BanglaHalluEval" / "Hallucination Generated Answers" / "qa_4000.csv"
TRAINSET_CONTEXT_PATH = DATA_DIR / "trainset_context.csv"

RNG_SEED = 42
# keep hallucinated:faithful roughly 1.5:1 (not the raw 4:1 in qa_4000.csv) and mix
# across all 4 hallucination patterns rather than any single one
HALLUCINATED_PER_FAITHFUL = 1.5


def load_benhallueval_pairs() -> pd.DataFrame:
    gt = pd.read_csv(GT_PATH).drop_duplicates(subset="id", keep="first")
    gen = pd.read_csv(GEN_PATH)

    faithful = gt.rename(columns={"question": "prompt_bn", "correct_answer": "response_bn"})
    faithful = faithful.assign(label=1, source_id=faithful["id"], origin="benhallueval_gt")
    faithful = faithful[["context", "prompt_bn", "response_bn", "label", "source_id", "origin"]]

    # only keep hallucinated rows whose source_id actually has a matching faithful row,
    # so every group has at least one positive and one negative example
    gen = gen[gen["source_id"].isin(faithful["source_id"])]

    rng = np.random.default_rng(RNG_SEED)
    n_hallucinated = min(len(gen), int(len(faithful) * HALLUCINATED_PER_FAITHFUL))
    # sample evenly across the 4 patterns (factualness/comprehension/specificity/inference)
    # so no single hallucination style dominates
    per_pattern = n_hallucinated // gen["pattern"].nunique()
    hallucinated = (
        gen.groupby("pattern", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_pattern), random_state=RNG_SEED))
    )
    hallucinated = hallucinated.rename(columns={"question": "prompt_bn", "hallucinated_answer": "response_bn"})
    hallucinated = hallucinated.assign(label=0, origin="benhallueval_gen")
    hallucinated = hallucinated[["context", "prompt_bn", "response_bn", "label", "source_id", "origin"]]

    pattern_mix = gen.loc[hallucinated.index, "pattern"].value_counts().to_dict()
    print(f"BenHalluEval: {len(faithful)} faithful, {len(hallucinated)} hallucinated "
          f"of {len(gen)} available, pattern mix: {pattern_mix}")
    return pd.concat([faithful, hallucinated], ignore_index=True)


def load_sample_real() -> pd.DataFrame:
    df = pd.read_csv(TRAINSET_CONTEXT_PATH)
    df = df[df["source"] == "sample-real"].copy()
    # gold rows from our own competition sample: give each its own source_id group
    # (never overlaps with BenHalluEval ids) so they can land on either side of the split
    df["source_id"] = "sample_real_" + df.index.astype(str)
    df["origin"] = "sample_real"
    print(f"trainset_context.csv sample-real: {len(df)} rows "
          f"({df['label'].mean():.1%} faithful)")
    return df[["context", "prompt_bn", "response_bn", "label", "source_id", "origin"]]


def group_split(df: pd.DataFrame, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by source_id so no passage's faithful/hallucinated variants are split apart."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=RNG_SEED)
    train_idx, val_idx = next(splitter.split(df, groups=df["source_id"]))
    train, val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    overlap = set(train["source_id"]) & set(val["source_id"])
    assert not overlap, f"leakage: {len(overlap)} source_ids in both splits"
    return train, val


def build() -> None:
    combined = pd.concat([load_benhallueval_pairs(), load_sample_real()], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)  # shuffle

    train, val = group_split(combined)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_path, val_path = CACHE_DIR / "lane2_train.csv", CACHE_DIR / "lane2_val.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

    for name, split in [("train", train), ("val", val)]:
        print(f"\n{name}: {len(split)} rows, {split['label'].mean():.1%} faithful, "
              f"{split['source_id'].nunique()} unique source_ids")
        print(split["origin"].value_counts().to_string())
    print(f"\nsaved {train_path} and {val_path}")


if __name__ == "__main__":
    build()
