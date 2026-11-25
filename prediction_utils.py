"""Small, transparent safeguards around the trained binary text classifier."""
from __future__ import annotations
import re

_NEUTRAL_GREETING = re.compile(
    r"^\s*(?:(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|how\s+are\s+you)"
    r"[\s,!.?;:-]*){1,4}(?:have\s+a\s+nice\s+day[!. ]*)?\s*$",
    re.IGNORECASE,
)
_SUSPICIOUS = re.compile(
    r"https?://|www\.|\b(?:password|verify|verification|urgent|wire transfer|"
    r"bank account|claim|winner|won|prize|gift card|crypto|bitcoin|investment|"
    r"limited time|click here|act now|payment required|account suspended)\b|\$\s*\d",
    re.IGNORECASE,
)


def classify_message(model, text: str) -> tuple[int, str]:
    """Return (0=HAM/1=SPAM, explanation). No hard-coded decision for risky text."""
    content = (text or "").strip()
    if not content:
        raise ValueError("Message is empty.")
    # Prevent a common false positive: plain conversational greetings.
    # Any link, money, account, urgency, or prize indicator bypasses this rule.
    if len(content.split()) <= 14 and _NEUTRAL_GREETING.fullmatch(content) and not _SUSPICIOUS.search(content):
        return 0, "Neutral conversational greeting; no obvious link or urgency indicator."
    prediction = int(model.predict([content])[0])
    return prediction, "Classified by the selected trained model."
