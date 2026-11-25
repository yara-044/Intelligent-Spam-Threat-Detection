"""Train and compare spam classifiers with leakage-safe train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load, validate, normalize, and de-duplicate the HAM/SPAM dataset."""
    frame = pd.read_csv(path)
    required = {"label", "label_num", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["label", "label_num", "text"]).copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label_num"] = pd.to_numeric(frame["label_num"], errors="coerce")
    frame = frame[
        frame["label"].isin(["ham", "spam"])
        & frame["label_num"].isin([0, 1])
        & frame["text"].ne("")
    ].copy()
    frame["label_num"] = frame["label_num"].astype(int)
    expected = frame["label"].map({"ham": 0, "spam": 1})
    if not expected.equals(frame["label_num"]):
        raise ValueError("label and label_num contain inconsistent values.")
    frame["normalized_text"] = (
        frame["text"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    conflicting = frame.groupby("normalized_text")["label_num"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Identical messages have conflicting labels. Resolve them before training.")
    frame = frame.drop_duplicates("normalized_text").reset_index(drop=True)
    if frame["label_num"].nunique() != 2 or frame["label_num"].value_counts().min() < 5:
        raise ValueError("Training requires at least 5 unique examples in each class.")
    return frame


def make_pipeline(classifier: object) -> Pipeline:
    """Shared word + character TF-IDF pipeline used by original and custom training."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=1, max_df=0.99,
            max_features=60000, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, strip_accents="unicode",
            ngram_range=(3, 5), min_df=2, max_df=0.995,
            max_features=100000, sublinear_tf=True,
        )),
    ], transformer_weights={"word": 1.0, "char": 0.8})
    return Pipeline([("features", features), ("classifier", classifier)])


def score_model(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(texts)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
    else:
        scores = predictions
    auc = roc_auc_score(labels, scores) if labels.nunique() == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(auc),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def tune_model(name: str, pipeline: Pipeline, parameter_grid: dict,
               texts: pd.Series, labels: pd.Series, folds: int):
    """Tune using training data only; never inspect validation/test here."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, parameter_grid, scoring="f1", cv=cv, n_jobs=2,
        refit=True, error_score="raise",
    )
    search.fit(texts, labels)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def _split_three(texts: pd.Series, labels: pd.Series):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y_train_val,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _candidate_models():
    return {
        "Multinomial Naive Bayes": (
            make_pipeline(MultinomialNB()),
            {"classifier__alpha": [0.1, 0.3, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            make_pipeline(LogisticRegression(
                max_iter=3000, class_weight="balanced",
                solver="liblinear", random_state=RANDOM_STATE,
            )),
            {"classifier__C": [0.5, 1.0, 2.0, 4.0]},
        ),
        "Linear SVM": (
            make_pipeline(LinearSVC(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
            {"classifier__C": [0.25, 0.5, 1.0, 2.0]},
        ),
    }


def train_custom_dataset(texts, labels):
    """Train custom CSV models with the same features and leakage-safe evaluation."""
    x = pd.Series(texts).fillna("").astype(str).str.strip()
    y = pd.Series(labels).astype(int)
    valid = x.ne("")
    x, y = x[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    normalized = x.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    temp = pd.DataFrame({"text": x, "label": y, "normalized": normalized})
    if (temp.groupby("normalized")["label"].nunique() > 1).any():
        raise ValueError("The custom dataset has identical messages with conflicting labels.")
    temp = temp.drop_duplicates("normalized").reset_index(drop=True)
    x, y = temp["text"], temp["label"]
    if y.nunique() != 2 or y.value_counts().min() < 10:
        raise ValueError("Please provide at least 10 unique messages for each class.")
    x_train, x_val, x_test, y_train, y_val, y_test = _split_three(x, y)
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough training examples for cross-validation.")

    tuned = {}
    validation_scores = {}
    details = {}
    for name, (pipeline, grid) in _candidate_models().items():
        fitted, cv_f1, params = tune_model(name, pipeline, grid, x_train, y_train, folds)
        tuned[name] = fitted
        validation_scores[name] = f1_score(y_val, fitted.predict(x_val), zero_division=0)
        details[name] = {"cv_f1": cv_f1, "validation_f1": float(validation_scores[name]),
                         "best_parameters": params}

    # Random Forest is retained as a comparison, not used as a tuning shortcut on test data.
    rf = make_pipeline(RandomForestClassifier(
        n_estimators=150, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=2,
    ))
    rf.fit(x_train, y_train)
    tuned["Random Forest"] = rf
    validation_scores["Random Forest"] = f1_score(y_val, rf.predict(x_val), zero_division=0)
    details["Random Forest"] = {"cv_f1": float("nan"),
                                "validation_f1": float(validation_scores["Random Forest"]),
                                "best_parameters": {}}

    best_name = max(validation_scores, key=validation_scores.get)
    # Refit the selected model on train + validation only; test remains untouched.
    chosen = tuned[best_name]
    chosen.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
    tuned[best_name] = chosen

    rows = []
    for name, candidate in tuned.items():
        # Other candidates are also refit on train+validation for fair final reporting.
        if name != best_name:
            candidate.fit(pd.concat([x_train, x_val]), pd.concat([y_train, y_val]))
        metrics = score_model(candidate, x_test, y_test)
        rows.append({
            "Model": name, "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"], "Recall": metrics["recall"],
            "F1-score": metrics["f1"], "FPR": metrics["false_positive_rate"],
            "Validation F1": validation_scores[name],
        })
    results = pd.DataFrame(rows).sort_values("Validation F1", ascending=False)
    return results, tuned, best_name, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.data)
    class_counts = frame["label"].value_counts().to_dict()
    x_train, x_val, x_test, y_train, y_val, y_test = _split_three(
        frame["text"], frame["label_num"]
    )
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough training examples for stratified cross-validation.")

    tuned, validation_scores, cv_details = {}, {}, {}
    for name, (pipeline, grid) in _candidate_models().items():
        print(f"Tuning {name} with {folds}-fold cross-validation on training data...")
        fitted, cv_f1, params = tune_model(name, pipeline, grid, x_train, y_train, folds)
        tuned[name] = fitted
        validation_scores[name] = f1_score(y_val, fitted.predict(x_val), zero_division=0)
        cv_details[name] = {
            "best_cv_f1": cv_f1, "validation_f1": float(validation_scores[name]),
            "best_parameters": params,
        }

    rf = make_pipeline(RandomForestClassifier(
        n_estimators=150, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=2,
    ))
    rf.fit(x_train, y_train)
    tuned["Random Forest"] = rf
    validation_scores["Random Forest"] = f1_score(y_val, rf.predict(x_val), zero_division=0)
    cv_details["Random Forest"] = {
        "best_cv_f1": float("nan"),
        "validation_f1": float(validation_scores["Random Forest"]),
        "best_parameters": {},
    }

    best_name = max(validation_scores, key=validation_scores.get)
    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])
    rows, matrices, fitted_models = [], {}, {}
    for name, candidate in tuned.items():
        candidate.fit(x_train_val, y_train_val)
        fitted_models[name] = candidate
        metrics = score_model(candidate, x_test, y_test)
        rows.append({"model": name, "cv_f1": cv_details[name]["best_cv_f1"],
                     "validation_f1": validation_scores[name], **metrics})
        predictions = candidate.predict(x_test)
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1])
        report = classification_report(
            y_test, predictions, labels=[0, 1], target_names=["HAM", "SPAM"],
            digits=4, zero_division=0,
        )
        (args.output / f"{name.lower().replace(' ', '_')}_report.txt").write_text(
            report, encoding="utf-8"
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_f1", "cv_f1"], ascending=False, na_position="last"
    )
    results.to_csv(args.output / "metrics.csv", index=False, float_format="%.6f")
    joblib.dump({"model": fitted_models[best_name], "name": best_name},
                args.output / "best_model.joblib")
    pd.DataFrame([
        {"class": "HAM", "count": int(class_counts.get("ham", 0))},
        {"class": "SPAM", "count": int(class_counts.get("spam", 0))},
    ]).to_csv(args.output / "class_distribution.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE, "records_after_cleaning": int(len(frame)),
        "training_records": int(len(x_train)), "validation_records": int(len(x_val)),
        "test_records": int(len(x_test)), "cross_validation_folds": int(folds),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "selection_metric": "Spam F1 on validation split; CV tuning uses training split only",
        "final_evaluation": "Untouched test split",
        "false_positive_rate": "FP / (FP + TN)",
        "best_model": best_name, "cross_validation": cv_details,
        "note": ("Word and character TF-IDF features are fitted only on training data. "
                 "The test split is not used for tuning or model selection. "
                 "Performance on future personal Gmail messages may differ from this dataset."),
    }
    (args.output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (name, matrix) in zip(axes.ravel(), matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["HAM", "SPAM"])
        axis.set_yticks([0, 1], ["HAM", "SPAM"])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, matrix[row, col], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    plot_columns = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc"]
    ax = results.set_index("model")[plot_columns].plot(
        kind="bar", figsize=(13, 6), ylim=(0, 1), rot=20
    )
    ax.set_ylabel("Score")
    ax.set_title("Classifier comparison (untouched test set)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(args.output / "model_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Validated unique records: {len(frame)}")
    print(f"Training / validation / test: {len(x_train)} / {len(x_val)} / {len(x_test)}")
    print(f"Cross-validation folds: {folds}")
    print(f"Best model selected by validation F1: {best_name}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
