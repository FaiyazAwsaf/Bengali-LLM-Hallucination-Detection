"""Track B — closed-book rows: independent regeneration + agreement features.

Core idea (implementation.md section 3): a Bengali-capable open-weight LLM answers
`prompt_bn` on its own, WITHOUT ever seeing `response_bn`. If the given response
disagrees with the independent answer, that's evidence of hallucination. We turn
"how much do they agree" into features for a small classifier (track_b_classifier.py)
instead of a brittle hard threshold.

Models (verified on the HF API 2026-07-08 — exist, open licenses, Bengali coverage):
- Generator: md-nishat-008/TigerLLM-9B-it (CC-BY-4.0, ungated). Loaded 4-bit on GPU
  because 9B fp16 (~18GB) doesn't fit a T4/P100. Override with TIGER_MODEL env var
  (e.g. md-nishat-008/TigerLLM-1B-it if 9B is too slow).
- Embeddings: sentence-transformers/LaBSE (Apache-2.0).
- NLI agreement check reuses Track A's mDeBERTa model.

Generation is cached incrementally (flushed every batch) so a Colab disconnect
never loses more than one batch of work.

Usage:
    python track_b_consistency.py gen train|test        # GPU, slow for test
    python track_b_consistency.py features train|test   # GPU preferred, fast
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import load_samples, load_test
from track_a_nli import CACHE_DIR, score_pairs

GEN_MODEL = os.environ.get("TIGER_MODEL", "md-nishat-008/TigerLLM-9B-it")
EMBED_MODEL = "sentence-transformers/LaBSE"

GEN_PROMPT = "নিম্নলিখিত প্রশ্নের সংক্ষিপ্ত এবং সঠিক উত্তর দিন।\n\nপ্রশ্ন: {question}"

FEATURE_COLUMNS = [
    "emb_cos", "nli_fwd", "nli_bwd", "tok_jaccard", "digit_agree",
    "len_response", "len_gen",
]


def gen_cache_path(split: str) -> Path:
    return CACHE_DIR / f"tigerllm_{split}_answers.csv"


def features_cache_path(split: str) -> Path:
    return CACHE_DIR / f"track_b_features_{split}.csv"


def load_split(split: str) -> pd.DataFrame:
    return load_samples() if split == "train" else load_test()


# --- Step 1: independent answer generation --------------------------------------------------

def generate_answers(df: pd.DataFrame, cache_path: Path, batch_size: int | None = None) -> pd.DataFrame:
    """Generate an answer for every closed-book prompt; resumable via the cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    closed = df[df["is_closed_book"]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(cache_path, index_col=0) if cache_path.exists() else pd.DataFrame(columns=["gen_answer"])
    todo = closed.loc[~closed.index.isin(done.index)]
    print(f"{len(closed)} closed-book rows, {len(done)} cached, {len(todo)} to generate ({GEN_MODEL})")
    if todo.empty:
        return done

    use_cuda = torch.cuda.is_available()
    if batch_size is None:
        batch_size = int(os.environ.get("TIGER_BATCH", 8 if use_cuda else 1))
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, padding_side="left")
    model_kwargs: dict = {"device_map": "auto"} if use_cuda else {}
    if use_cuda:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, **model_kwargs).eval()

    start = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo.iloc[i : i + batch_size]
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": GEN_PROMPT.format(question=q)}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in batch["prompt_bn"].astype(str)
        ]
        inputs = tokenizer(chats, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        answers = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        done = pd.concat([done, pd.DataFrame({"gen_answer": [a.strip() for a in answers]}, index=batch.index)])
        done.to_csv(cache_path)  # flush every batch — resumable after a disconnect
        n = len(done)
        rate = (time.time() - start) / max(n - (len(closed) - len(todo)), 1)
        print(f"  {n}/{len(closed)} answers ({rate:.1f}s/row)", flush=True)
    return done


# --- Step 2: agreement features --------------------------------------------------------------

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", str(text).translate(BN_DIGITS)))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wঀ-৿]+", str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def embed_cosine(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    emb_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False)
    return (emb_a * emb_b).sum(axis=1)


def build_features(df: pd.DataFrame, gen: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Agreement features for every closed-book row, aligned to df.index."""
    closed = df[df["is_closed_book"]]
    missing = closed.index.difference(gen.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} closed-book rows lack generated answers — run `gen` first")

    responses = closed["response_bn"].astype(str).tolist()
    answers = gen.loc[closed.index, "gen_answer"].fillna("").astype(str).tolist()

    feats = pd.DataFrame(index=closed.index)
    feats["emb_cos"] = embed_cosine(answers, responses)
    # NLI in both directions: does the response follow from the independent answer, and vice versa
    fwd = score_pairs(answers, responses)
    bwd = score_pairs(responses, answers)
    feats["nli_fwd"] = (fwd["p_entail"] - fwd["p_contra"]).to_numpy()
    feats["nli_bwd"] = (bwd["p_entail"] - bwd["p_contra"]).to_numpy()
    feats["tok_jaccard"] = [_jaccard(_tokens(a), _tokens(r)) for a, r in zip(answers, responses)]
    feats["digit_agree"] = [_jaccard(_digits(a), _digits(r)) for a, r in zip(answers, responses)]
    feats["len_response"] = np.log1p([len(r) for r in responses])
    feats["len_gen"] = np.log1p([len(a) for a in answers])

    out = feats.reindex(df.index)  # context rows -> NaN, same convention as the NLI cache
    out.to_csv(out_path)
    print(f"saved {out_path} ({len(closed)} closed-book rows featurized)")
    return out


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("gen", "features") or sys.argv[2] not in ("train", "test"):
        sys.exit(__doc__)
    step, split = sys.argv[1], sys.argv[2]
    df = load_split(split)
    if step == "gen":
        generate_answers(df, gen_cache_path(split))
    else:
        gen = pd.read_csv(gen_cache_path(split), index_col=0)
        build_features(df, gen, features_cache_path(split))


if __name__ == "__main__":
    main()
