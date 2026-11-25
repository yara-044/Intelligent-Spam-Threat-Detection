"""Shared message preparation and prediction helpers."""
from __future__ import annotations


def prepare_email_text(subject: str = "", body: str = "") -> str:
    """Use the same subject/body representation for every Gmail message."""
    subject = (subject or "").strip()
    body = (body or "").strip()
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body:
        parts.append(body)
    return "\n\n".join(parts).strip()


def classify_message(model, text: str) -> tuple[int, str]:
    """Return (0=HAM, 1=SPAM); all decisions come from the trained model."""
    content = (text or "").strip()
    if not content:
        raise ValueError("Message is empty.")
    prediction = int(model.predict([content])[0])
    return prediction, "Classified by the selected trained machine-learning model."


def classify_email(model, subject: str = "", body: str = "") -> tuple[int, str]:
    """Classify a Gmail message using its subject and body, not sender heuristics."""
    return classify_message(model, prepare_email_text(subject, body))
