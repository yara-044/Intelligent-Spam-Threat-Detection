"""Train and compare duplicate-safe text classifiers for spam detection."""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"label", "label_num", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=["label", "label_num", "text"]).copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str)
    frame["label_num"] = pd.to_numeric(frame["label_num"], errors="raise").astype(int)
    expected = frame["label"].map({"ham": 0, "spam": 1})
    if expected.isna().any() or not expected.equals(frame["label_num"]):
        raise ValueError("label and label_num contain inconsistent values")

    frame["normalized_text"] = (
        frame["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    frame = frame.drop_duplicates(subset=["normalized_text"], keep="first")
    return frame


def make_pipeline(classifier: object) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def score_model(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(texts)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    else:
        scores = model.decision_function(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.data)
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        frame["text"],
        frame["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["label_num"],
    )

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    rows = []
    matrices = {}
    fitted = {}
    for name, classifier in models.items():
        model = make_pipeline(classifier)
        model.fit(texts_train, labels_train)
        metrics = score_model(model, texts_test, labels_test)
        rows.append({"model": name, **metrics})
        matrices[name] = confusion_matrix(labels_test, model.predict(texts_test))
        fitted[name] = model
        report = classification_report(labels_test, model.predict(texts_test), digits=4)
        (args.output / f"{name.lower().replace(' ', '_')}_report.txt").write_text(report)

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results.to_csv(args.output / "metrics.csv", index=False, float_format="%.6f")
    best_name = str(results.iloc[0]["model"])
    joblib.dump({"model": fitted[best_name], "name": best_name}, args.output / "best_model.joblib")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (name, matrix) in zip(axes.ravel(), matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["ham", "spam"])
        axis.set_yticks([0, 1], ["ham", "spam"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    ax = results.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(12, 6), ylim=(0, 1), rot=20
    )
    ax.set_ylabel("Score")
    ax.set_title("Spam classifier comparison")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(args.output / "model_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Validated records: {len(frame)}")
    print(f"Training records: {len(texts_train)}")
    print(f"Test records: {len(texts_test)}")
    print(f"Best model by spam F1: {best_name}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()