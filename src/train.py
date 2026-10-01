#!/usr/bin/env python3
"""Handwritten digit classification on sklearn digits (0-9).

Reproducible stratified 60/20/20 split (seed 42). Compare Dummy baseline,
StandardScaler+LogisticRegression, and RandomForest on validation macro-F1;
refit winner on train+val; evaluate once on held-out test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X, y = digits.data, digits.target
    # Preserve original sample indices for test_predictions.csv
    indices = list(range(len(y)))

    # First split: 60% train, 40% temp (stratified)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X,
        y,
        indices,
        test_size=0.4,
        random_state=SEED,
        stratify=y,
    )
    # Second split: 20% val, 20% test from temp (stratified) => overall 60/20/20
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    split_counts = {
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_frac": round(len(y_train) / len(y), 4),
        "val_frac": round(len(y_val) / len(y), 4),
        "test_frac": round(len(y_test) / len(y), 4),
        "random_state": SEED,
    }

    candidates = {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    val_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        val_results[name] = metrics_dict(y_val, pred_val)
        print(f"[val] {name}: {val_results[name]}")

    # Select by validation macro-F1
    chosen = max(val_results.keys(), key=lambda k: val_results[k]["macro_f1"])
    print(f"[select] chosen_model={chosen} by val macro-F1")

    # Rebuild fresh final model (same hyperparams) and fit on train+val
    final_builders = {
        "dummy": lambda: DummyClassifier(strategy="most_frequent", random_state=SEED),
        "logreg": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    final_model = final_builders[chosen]()

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    t0 = time.perf_counter()
    final_model.fit(X_trainval, y_trainval)
    train_time_sec = time.perf_counter() - t0

    y_pred = final_model.predict(X_test)
    test_metrics = metrics_dict(y_test, y_pred)
    test_metrics["train_time_sec"] = float(train_time_sec)
    test_metrics["chosen_model"] = chosen

    print(f"[test] {test_metrics}")

    out = {
        "split_counts": split_counts,
        "val_metrics": val_results,
        "chosen_model": chosen,
        "selection_criterion": "val_macro_f1",
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        },
        "final_train_time_sec": float(train_time_sec),
        "notes": (
            "Preprocessors/models fit on train only during candidate comparison; "
            "final model refit on train+val; test used once for reporting."
        ),
    }

    metrics_path = RESULTS / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[write] {metrics_path}")

    pred_df = pd.DataFrame(
        {
            "sample_index": idx_test,
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    pred_path = RESULTS / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[write] {pred_path}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
