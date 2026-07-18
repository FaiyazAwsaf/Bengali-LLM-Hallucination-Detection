"""Lane 2 — fine-tune BanglaBERT as a context/response cross-encoder.

Input format: [CLS] context [SEP] prompt_bn + " " + response_bn [SEP]
Output: P(hallucinated) via a binary classification head on top of [CLS].

This is a small encoder (~110M params) fine-tuned with a low learning rate, dropout,
and early stopping — implementation.md's explicit condition for fine-tuning being
acceptable at this data scale (never fine-tune a large generative model on this little
labeled data). Training data: src/lane2_dataset.py's leak-free, contrastive-paired split
(BenHalluEval same-context right/wrong answers + our own gold sample-real rows).

Model: csebuetnlp/banglabert (verified on the HF API 2026-07-18: exists, ungated,
weights present).

Only ships as Lane 2 / Track A's replacement if it clearly beats the current zero-shot
NLI's 0.683 CV Macro-F1 on the context subset (src/track_a_nli.py) — that comparison
happens in lane2_predict.py via the existing validation.py harness, not here. This
script's job is only to produce a checkpoint; it is not the go/no-go decision.

Usage:
    python src/lane2_finetune.py
Output: models/lane2_banglabert/ (checkpoint + tokenizer)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "csebuetnlp/banglabert"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "lane2_banglabert"
MAX_LENGTH = 384  # context median 546 chars but BanglaBERT's WordPiece tokens run
                   # shorter than raw chars for Bengali; 384 tokens covers most rows
                   # without ballooning train time - checked against p90 in __main__


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        hypothesis = df["prompt_bn"].astype(str) + " " + df["response_bn"].astype(str)
        self.encodings = tokenizer(
            df["context"].astype(str).tolist(),
            hypothesis.tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def main() -> None:
    train_df = pd.read_csv(CACHE_DIR / "lane2_train.csv")
    val_df = pd.read_csv(CACHE_DIR / "lane2_val.csv")
    print(f"train: {len(train_df)} rows | val: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds = PairDataset(train_df, tokenizer)
    val_ds = PairDataset(val_df, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=int(os.environ.get("LANE2_EPOCHS", 6)),
        per_device_train_batch_size=int(os.environ.get("LANE2_BATCH", 16)),
        per_device_eval_batch_size=32,
        learning_rate=float(os.environ.get("LANE2_LR", 1e-5)),  # small LR: tiny data, avoid overfit
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nfinal val Macro-F1: {metrics['eval_macro_f1']:.3f}")
    train_metrics = trainer.evaluate(train_ds)
    print(f"train Macro-F1 (for overfit check): {train_metrics['eval_macro_f1']:.3f}")
    if train_metrics["eval_macro_f1"] - metrics["eval_macro_f1"] > 0.15:
        print("WARNING: train >> val Macro-F1 — looks like overfitting, treat this run with suspicion")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))
    print(f"saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
