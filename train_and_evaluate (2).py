"""Train and compare robust text classifiers for spam detection."""

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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
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
        frame["text"]
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    conflicting = frame.groupby("normalized_text")["label_num"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            "The dataset contains identical messages with conflicting labels. "
            "Resolve these rows before training."
        )

    frame = frame.drop_duplicates(subset=["normalized_text"], keep="first")
    frame = frame[frame["normalized_text"].ne("")].reset_index(drop=True)

    if frame["label_num"].nunique() != 2:
        raise ValueError("Training requires both HAM (0) and SPAM (1) examples.")
    if frame["label_num"].value_counts().min() < 5:
        raise ValueError("At least 5 unique examples per class are required.")

    return frame


def make_pipeline(classifier: object) -> Pipeline:
    """Build a text pipeline; fit TF-IDF only on the training data."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        max_features=75000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def score_model(
    model: Pipeline, texts: pd.Series, labels: pd.Series
) -> dict[str, float]:
    predictions = model.predict(texts)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
    else:
        scores = predictions

    auc = float("nan")
    if labels.nunique() == 2:
        auc = roc_auc_score(labels, scores)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": auc,
    }


def tune_model(
    name: str,
    pipeline: Pipeline,
    parameter_grid: dict,
    texts: pd.Series,
    labels: pd.Series,
    folds: int,
) -> tuple[Pipeline, float, dict]:
    """Tune supported models using training data only."""
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(texts, labels)
    return search.best_estimator_, float(search.best_score_), search.best_params_


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

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        frame["text"],
        frame["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["label_num"],
    )

    min_train_class = int(labels_train.value_counts().min())
    folds = min(5, min_train_class)
    if folds < 2:
        raise ValueError("Not enough training examples for stratified cross-validation.")

    candidates = {
        "Multinomial Naive Bayes": (
            make_pipeline(MultinomialNB()),
            {"classifier__alpha": [0.1, 0.3, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            make_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
            ),
            {"classifier__C": [0.5, 1.0, 2.0, 4.0]},
        ),
        "Linear SVM": (
            make_pipeline(
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            {"classifier__C": [0.25, 0.5, 1.0, 2.0]},
        ),
    }

    rows = []
    matrices = {}
    fitted = {}
    cv_details = {}

    for name, (pipeline, grid) in candidates.items():
        print(f"Tuning {name} with {folds}-fold stratified cross-validation...")
        model, cv_f1, best_params = tune_model(
            name, pipeline, grid, texts_train, labels_train, folds
        )
        metrics = score_model(model, texts_test, labels_test)
        rows.append(
            {"model": name, "cv_f1": cv_f1, **metrics}
        )
        predictions = model.predict(texts_test)
        matrices[name] = confusion_matrix(
            labels_test, predictions, labels=[0, 1]
        )
        fitted[name] = model
        cv_details[name] = {
            "best_cv_f1": cv_f1,
            "best_parameters": best_params,
        }
        report = classification_report(
            labels_test,
            predictions,
            labels=[0, 1],
            target_names=["HAM", "SPAM"],
            digits=4,
            zero_division=0,
        )
        (args.output / f"{name.lower().replace(' ', '_')}_report.txt").write_text(
            report, encoding="utf-8"
        )

    # Keep Random Forest as a comparison model, but avoid an unnecessarily
    # expensive large parameter search for sparse text features.
    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    print("Training Random Forest...")
    rf.fit(texts_train, labels_train)
    rf_metrics = score_model(rf, texts_test, labels_test)
    rf_predictions = rf.predict(texts_test)
    rows.append({"model": "Random Forest", "cv_f1": float("nan"), **rf_metrics})
    matrices["Random Forest"] = confusion_matrix(
        labels_test, rf_predictions, labels=[0, 1]
    )
    fitted["Random Forest"] = rf
    (args.output / "random_forest_report.txt").write_text(
        classification_report(
            labels_test,
            rf_predictions,
            labels=[0, 1],
            target_names=["HAM", "SPAM"],
            digits=4,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    results = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "recall"], ascending=False
    )
    results.to_csv(args.output / "metrics.csv", index=False, float_format="%.6f")

    best_name = str(results.iloc[0]["model"])
    joblib.dump(
        {"model": fitted[best_name], "name": best_name},
        args.output / "best_model.joblib",
    )

    pd.DataFrame(
        [
            {"class": "HAM", "count": int(class_counts.get("ham", 0))},
            {"class": "SPAM", "count": int(class_counts.get("spam", 0))},
        ]
    ).to_csv(args.output / "class_distribution.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "records_after_cleaning": int(len(frame)),
        "training_records": int(len(texts_train)),
        "test_records": int(len(texts_test)),
        "cross_validation_folds": int(folds),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "selection_metric": "Spam-class F1 on the untouched holdout test set",
        "best_model": best_name,
        "cross_validation": cv_details,
        "note": (
            "Holdout metrics are measured on data not used for fitting or tuning. "
            "They estimate performance on data similar to this dataset and do not "
            "guarantee the same accuracy on future emails."
        ),
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
            for column in range(2):
                axis.text(
                    column, row, matrix[row, column],
                    ha="center", va="center"
                )
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    plot_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = results.set_index("model")[plot_columns].plot(
        kind="bar", figsize=(12, 6), ylim=(0, 1), rot=20
    )
    ax.set_ylabel("Score")
    ax.set_title("Spam classifier comparison (holdout test set)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(args.output / "model_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Validated unique records: {len(frame)}")
    print(f"Training records: {len(texts_train)}")
    print(f"Test records: {len(texts_test)}")
    print(f"Cross-validation folds: {folds}")
    print(f"Best model by holdout spam F1: {best_name}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
