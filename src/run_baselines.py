"""Put the naive baselines + the TF-IDF starter baseline on the board.

Every future model must beat these by more than one fold-std (implementation.md
section 5). The TF-IDF number doubles as a sanity check that the harness roughly
reproduces the public starter notebook's ~0.546 Macro-F1.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data_loading import load_samples
from validation import NAIVE_BASELINES, log_experiment, run_cv


def tfidf_logreg(train_df, eval_df):
    """Char n-gram TF-IDF on prompt+response, as in the public starter notebook."""
    text = lambda df: (df["prompt_bn"].astype(str) + " " + df["response_bn"].astype(str)).to_numpy()
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000),
        LogisticRegression(max_iter=1000),
    )
    model.fit(text(train_df), train_df["label"])
    return np.asarray(model.predict(text(eval_df)))


def main() -> None:
    df = load_samples()
    candidates = dict(NAIVE_BASELINES)
    candidates["tfidf_char_logreg"] = tfidf_logreg
    for name, fit_predict in candidates.items():
        result = run_cv(df, fit_predict, name=name)
        result.print_report()
        log_experiment(result, notes="milestone 0 baseline")


if __name__ == "__main__":
    main()
