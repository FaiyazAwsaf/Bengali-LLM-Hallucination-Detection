"""Track A — zero-shot NLI entailment for context-present rows.

Premise = context, hypothesis = response_bn. Decision score is
P(entailment) - P(contradiction), binarized with a threshold that is calibrated
on each CV training fold only (never on the fold being scored — implementation.md 2.3).

Closed-book rows have no context to entail from; they get the majority-class
prediction here so the combined Macro-F1 is comparable with the baselines. Track B
replaces that part later.

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
(verified 2026-07-08 on the HF API: exists, MIT license, Bengali listed in languages).

Scores are cached to data/cache/ so CV and submission building never recompute them.
Usage:
    python src/track_a_nli.py cv       # score train rows if needed, run repeated CV
    python src/track_a_nli.py test     # score test-set context rows (slow on CPU)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data_loading import load_samples, load_test
from validation import log_experiment, run_cv

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
TRAIN_SCORES = CACHE_DIR / "nli_train_scores.csv"
TEST_SCORES = CACHE_DIR / "nli_test_scores.csv"

MAJORITY_CLOSED_BOOK = 0  # majority label among closed-book train rows (47.3% faithful)


def score_pairs(premises: list[str], hypotheses: list[str], batch_size: int | None = None) -> pd.DataFrame:
    """Return P(entailment/neutral/contradiction) per pair."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = int(os.environ.get("NLI_BATCH", 32 if device == "cuda" else 8))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    label_order = [model.config.label2id[k] for k in ("entailment", "neutral", "contradiction")]

    probs = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(premises), batch_size):
            batch = tokenizer(
                premises[i : i + batch_size],
                hypotheses[i : i + batch_size],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(device)
            p = torch.softmax(model(**batch).logits, dim=-1)[:, label_order]
            probs.append(p.cpu().numpy())
            done = min(i + batch_size, len(premises))
            print(f"  scored {done}/{len(premises)} ({time.time() - start:.0f}s)", flush=True)
    return pd.DataFrame(np.vstack(probs), columns=["p_entail", "p_neutral", "p_contra"])


def scores_for(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """NLI scores aligned to df.index; closed-book rows get NaN. Cached on disk."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)
        if len(cached) == len(df):
            return cached
    ctx = df[~df["is_closed_book"]]
    scored = score_pairs(ctx["context"].astype(str).tolist(), ctx["response_bn"].astype(str).tolist())
    scored.index = ctx.index
    out = scored.reindex(df.index)
    out["nli_score"] = out["p_entail"] - out["p_contra"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path)
    return out


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold on nli_score maximizing Macro-F1 (label 1 iff score >= t)."""
    best_t, best_f1 = 0.0, -1.0
    for t in np.linspace(-1, 1, 201):
        f1 = f1_score(labels, (scores >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def make_fit_predict(all_scores: pd.DataFrame):
    """Harness fit_predict: calibrate threshold on the train fold's context rows."""

    def fit_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        train_ctx = train_df[~train_df["is_closed_book"]]
        t = best_threshold(
            all_scores.loc[train_ctx.index, "nli_score"].to_numpy(),
            train_ctx["label"].to_numpy(),
        )
        preds = np.full(len(eval_df), MAJORITY_CLOSED_BOOK)
        ctx_mask = (~eval_df["is_closed_book"]).to_numpy()
        eval_scores = all_scores.loc[eval_df.index[ctx_mask], "nli_score"].to_numpy()
        preds[ctx_mask] = (eval_scores >= t).astype(int)
        return preds

    return fit_predict


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if mode == "cv":
        df = load_samples()
        scores = scores_for(df, TRAIN_SCORES)
        result = run_cv(df, make_fit_predict(scores), name="nli_zeroshot_ctx+majority_cb")
        result.print_report()
        log_experiment(result, notes=f"track A zero-shot {MODEL_NAME}, per-fold threshold")
        # reference: threshold fitted on all 130 context rows (used for the submission)
        ctx = df[~df["is_closed_book"]]
        t = best_threshold(scores.loc[ctx.index, "nli_score"].to_numpy(), ctx["label"].to_numpy())
        print(f"\nfull-train calibrated threshold: {t:.3f}")
    elif mode == "test":
        scores_for(load_test(), TEST_SCORES)
        print(f"saved {TEST_SCORES}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
