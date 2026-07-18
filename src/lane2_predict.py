"""Lane 2 — score context rows with the fine-tuned BanglaBERT cross-encoder, and run
it through the SAME repeated-CV harness used for Track A so the comparison is apples
to apples.

This is the actual go/no-go step: Lane 2 only replaces Track A's zero-shot NLI if it
clearly beats 0.683 CV Macro-F1 on our 130 real context rows (not on BenHalluEval val,
which is a proxy, not the competition's own labels).

Usage:
    python src/lane2_predict.py cv     # score our 130 context rows, run repeated CV
    python src/lane2_predict.py test   # score test-set context rows (for submission)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loading import load_samples, load_test
from lane2_finetune import MAX_LENGTH, MODEL_OUT
from validation import log_experiment, run_cv

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
TRAIN_SCORES = CACHE_DIR / "lane2_train_scores.csv"
TEST_SCORES = CACHE_DIR / "lane2_test_scores.csv"
MAJORITY_CLOSED_BOOK = 0


def score_pairs(contexts: list[str], hypotheses: list[str], model_dir=MODEL_OUT,
                batch_size: int | None = None) -> pd.DataFrame:
    """P(hallucinated) for each (context, prompt+response) pair."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        batch_size = 32 if device == "cuda" else 8
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    probs = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = tokenizer(
                contexts[i : i + batch_size],
                hypotheses[i : i + batch_size],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            p = torch.softmax(model(**batch).logits, dim=-1)[:, 1]  # P(label=1 / hallucinated)
            probs.append(p.cpu().numpy())
            done = min(i + batch_size, len(contexts))
            print(f"  scored {done}/{len(contexts)} ({time.time() - start:.0f}s)", flush=True)
    return pd.DataFrame({"p_hallucinated": np.concatenate(probs)})


def scores_for(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """P(hallucinated) aligned to df.index; closed-book rows get NaN. Cached on disk."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)
        if len(cached) == len(df):
            return cached
    ctx = df[~df["is_closed_book"]]
    hyp = (ctx["prompt_bn"].astype(str) + " " + ctx["response_bn"].astype(str)).tolist()
    scored = score_pairs(ctx["context"].astype(str).tolist(), hyp)
    scored.index = ctx.index
    out = scored.reindex(df.index)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path)
    return out


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Threshold on P(hallucinated) maximizing Macro-F1 (label 0 iff prob >= t)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0, 1, 201):
        preds = (probs < t).astype(int)  # prob < t -> faithful (1), else hallucinated (0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def make_fit_predict(all_scores: pd.DataFrame):
    def fit_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        train_ctx = train_df[~train_df["is_closed_book"]]
        t = best_threshold(
            all_scores.loc[train_ctx.index, "p_hallucinated"].to_numpy(),
            train_ctx["label"].to_numpy(),
        )
        preds = np.full(len(eval_df), MAJORITY_CLOSED_BOOK)
        ctx_mask = (~eval_df["is_closed_book"]).to_numpy()
        eval_probs = all_scores.loc[eval_df.index[ctx_mask], "p_hallucinated"].to_numpy()
        preds[ctx_mask] = (eval_probs < t).astype(int)
        return preds

    return fit_predict


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if mode == "cv":
        df = load_samples()
        scores = scores_for(df, TRAIN_SCORES)
        result = run_cv(df, make_fit_predict(scores), name="lane2_banglabert_ctx+majority_cb")
        result.print_report()
        log_experiment(result, notes=f"Lane 2 fine-tuned {MODEL_OUT.name}, per-fold threshold")
        ctx = df[~df["is_closed_book"]]
        t = best_threshold(scores.loc[ctx.index, "p_hallucinated"].to_numpy(), ctx["label"].to_numpy())
        print(f"\nfull-train calibrated threshold: {t:.3f}")
        print("compare context Macro-F1 above against Track A's 0.683 (src/track_a_nli.py) "
              "before deciding whether Lane 2 replaces Track A.")
    elif mode == "test":
        scores_for(load_test(), TEST_SCORES)
        print(f"saved {TEST_SCORES}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
