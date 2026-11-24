from __future__ import annotations

import hashlib
import hmac
import secrets
import base64
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser

import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def create_gmail_flow(state: str | None = None, code_verifier: str | None = None):
    gmail_config = st.secrets["gmail"]
    client_config = {
        "web": {
            "client_id": gmail_config["client_id"],
            "client_secret": gmail_config["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [gmail_config["redirect_uri"]],
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        state=state,
        code_verifier=code_verifier,
    )
    flow.redirect_uri = gmail_config["redirect_uri"]
    return flow


def _oauth_secret() -> str:
    secret = st.secrets.get("GMAIL_OAUTH_STATE_SECRET")
    if not secret:
        raise ValueError("GMAIL_OAUTH_STATE_SECRET is missing from Streamlit secrets.")
    return str(secret)


def get_gmail_authorization_url() -> str:
    secret = _oauth_secret()
    nonce = secrets.token_urlsafe(32)
    signature = hmac.new(
        secret.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    signed_state = f"{nonce}.{signature}"
    code_verifier = hmac.new(
        secret.encode("utf-8"), f"pkce:{nonce}".encode("utf-8"), hashlib.sha256
    ).hexdigest()

    flow = create_gmail_flow(state=signed_state, code_verifier=code_verifier)
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return authorization_url


def complete_gmail_authorization(code: str, returned_state: str):
    if not code:
        raise ValueError("Google authorization code is missing.")
    if not returned_state or "." not in returned_state:
        raise ValueError("Invalid OAuth state.")

    nonce, received_signature = returned_state.rsplit(".", 1)
    secret = _oauth_secret()
    expected_signature = hmac.new(
        secret.encode("utf-8"), nonce.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(received_signature, expected_signature):
        raise ValueError("Invalid OAuth state signature.")

    code_verifier = hmac.new(
        secret.encode("utf-8"), f"pkce:{nonce}".encode("utf-8"), hashlib.sha256
    ).hexdigest()

    flow = create_gmail_flow(
        state=returned_state,
        code_verifier=code_verifier,
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials
    st.session_state["gmail_credentials"] = credentials
    return credentials


def get_gmail_account_email(service=None) -> str:
    """Return the email address of the currently connected Gmail mailbox."""
    service = service or get_gmail_service()
    if service is None:
        raise ValueError("Gmail session expired. Please connect Gmail again.")

    profile = service.users().getProfile(userId="me").execute()
    email_address = str(profile.get("emailAddress", "")).strip().lower()
    if not email_address:
        raise ValueError("Google did not return the connected Gmail address.")
    return email_address


def get_gmail_account_email(service=None) -> str:
    """Return the email address of the currently connected Gmail mailbox."""
    service = service or get_gmail_service()
    if service is None:
        raise ValueError("Gmail session expired. Please connect Gmail again.")
    profile = service.users().getProfile(userId="me").execute()
    email_address = str(profile.get("emailAddress", "")).strip().lower()
    if not email_address:
        raise ValueError("Google did not return the connected Gmail address.")
    return email_address


def get_gmail_service():
    credentials = st.session_state.get("gmail_credentials")
    if not credentials:
        return None

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            from google.auth.transport.requests import Request

            credentials.refresh(Request())
            st.session_state["gmail_credentials"] = credentials
        else:
            return None

    return build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False,
    )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self._skip_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(
            line.strip()
            for line in "".join(self._parts).splitlines()
            if line.strip()
        )


def _decode_part(part) -> str:
    try:
        content = part.get_content()
        return content if isinstance(content, str) else str(content)
    except (LookupError, UnicodeDecodeError, TypeError):
        payload = part.get_payload(decode=True)
        if not payload:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")


def decode_email_body(payload) -> str:
    """Extract plain text; fall back to readable text from HTML-only messages."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part):
        if part.is_multipart():
            for child in part.iter_parts():
                walk(child)
            return

        if part.get_content_disposition() == "attachment":
            return

        content_type = part.get_content_type().lower()
        if content_type == "text/plain":
            text = _decode_part(part).strip()
            if text:
                plain_parts.append(text)
        elif content_type == "text/html":
            html = _decode_part(part).strip()
            if html:
                html_parts.append(html)

    walk(payload)

    if plain_parts:
        return "\n\n".join(plain_parts)

    if html_parts:
        parser = _HTMLTextExtractor()
        for html in html_parts:
            parser.feed(html)
        parser.close()
        return parser.text()

    return ""


def _decode_gmail_raw(raw_data: str) -> bytes:
    if not raw_data:
        raise ValueError("Gmail returned an empty raw message.")
    padded = raw_data + "=" * (-len(raw_data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def fetch_latest_emails(max_results: int = 20) -> list[dict]:
    """Fetch recent messages and calculate SHA-256 over original RFC822 bytes."""
    service = get_gmail_service()
    if service is None:
        raise ValueError("Gmail session expired. Please connect Gmail again.")

    max_results = max(1, min(int(max_results), 100))
    response = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results)
        .execute()
    )

    emails = []
    for message in response.get("messages", []):
        details = (
            service.users()
            .messages()
            .get(userId="me", id=message["id"], format="raw")
            .execute()
        )
        raw_bytes = _decode_gmail_raw(details.get("raw", ""))
        parsed = BytesParser(policy=policy.default).parsebytes(raw_bytes)

        sender_header = str(parsed.get("From", ""))
        sender_name, sender_address = parseaddr(sender_header)
        body = decode_email_body(parsed)

        emails.append(
            {
                "id": message["id"],
                "sender": sender_address or sender_header,
                "sender_name": sender_name,
                "subject": str(parsed.get("Subject", "No Subject")),
                "date": str(parsed.get("Date", "")),
                "body": body,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "hash_algorithm": "SHA-256",
                "hash_scope": "Original Gmail RFC822 message bytes",
                "size_bytes": len(raw_bytes),
            }
        )

    return emails
