from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from html.parser import HTMLParser

import streamlit as st
from cryptography.fernet import Fernet, InvalidToken
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CONNECTIONS_TABLE = "gmail_connections"


def _fernet() -> Fernet:
    key = st.secrets.get("GMAIL_TOKEN_ENCRYPTION_KEY")
    if not key:
        raise ValueError("GMAIL_TOKEN_ENCRYPTION_KEY is missing from Streamlit secrets.")
    try:
        return Fernet(str(key).encode("utf-8"))
    except Exception as exc:
        raise ValueError("GMAIL_TOKEN_ENCRYPTION_KEY must be a valid Fernet key.") from exc


def create_gmail_flow(state: str | None = None, code_verifier: str | None = None):
    cfg = st.secrets["gmail"]
    client_config = {"web": {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [cfg["redirect_uri"]],
    }}
    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        state=state,
        code_verifier=code_verifier,
        autogenerate_code_verifier=False,
    )
    flow.redirect_uri = cfg["redirect_uri"]
    return flow


def _oauth_secret() -> str:
    value = st.secrets.get("GMAIL_OAUTH_STATE_SECRET")
    if not value:
        raise ValueError("GMAIL_OAUTH_STATE_SECRET is missing from Streamlit secrets.")
    return str(value)


def _encode_oauth_state(user_id: str, code_verifier: str) -> str:
    """Create signed state and securely carry the PKCE verifier across the redirect."""
    if not user_id:
        raise ValueError("Please sign in before connecting Gmail.")
    if not code_verifier:
        raise ValueError("Could not create the Gmail security verifier. Please try again.")
    encrypted_verifier = _fernet().encrypt(code_verifier.encode("utf-8")).decode("ascii")
    payload = {
        "uid": str(user_id),
        "nonce": secrets.token_urlsafe(32),
        "iat": __import__("time").time(),
        "pkce": encrypted_verifier,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(_oauth_secret().encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _decode_oauth_state(returned_state: str) -> dict:
    """Verify OAuth state without relying on Streamlit session_state surviving redirects."""
    if not returned_state or "." not in returned_state:
        raise ValueError("Google authorization response is missing its security state.")
    encoded, received_signature = returned_state.rsplit(".", 1)
    expected_signature = hmac.new(
        _oauth_secret().encode(), encoded.encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(received_signature, expected_signature):
        raise ValueError("Google authorization state is invalid. Please connect Gmail again.")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(raw.decode("utf-8"))
        issued_at = float(payload["iat"])
        user_id = str(payload["uid"]).strip()
        age = __import__("time").time() - issued_at
    except Exception as exc:
        raise ValueError("Google authorization state is malformed. Please connect Gmail again.") from exc
    if not user_id or age < -60 or age > 600:
        raise ValueError("Google authorization request expired. Please connect Gmail again.")
    encrypted_verifier = payload.get("pkce")
    if not encrypted_verifier:
        raise ValueError("The Gmail security verifier is missing. Please connect Gmail again.")
    try:
        payload["code_verifier"] = _fernet().decrypt(
            encrypted_verifier.encode("ascii")
        ).decode("utf-8")
    except Exception as exc:
        raise ValueError("The Gmail security verifier could not be restored. Please reconnect Gmail.") from exc
    return payload


def get_gmail_oauth_user_id(returned_state: str) -> str:
    """Return the app user id embedded in a valid signed OAuth state."""
    return str(_decode_oauth_state(returned_state)["uid"])


def get_gmail_authorization_url(user_id: str) -> str:
    code_verifier = secrets.token_urlsafe(64)
    state = _encode_oauth_state(user_id, code_verifier)
    flow = create_gmail_flow(state=state, code_verifier=code_verifier)
    url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="select_account consent",
    )
    return url


def _credentials_from_code(code: str, returned_state: str):
    if not code:
        raise ValueError("Google authorization response is incomplete.")
    state_payload = _decode_oauth_state(returned_state)
    flow = create_gmail_flow(
        state=returned_state,
        code_verifier=state_payload["code_verifier"],
    )
    # Pass the verifier explicitly to the token exchange as well as the
    # OAuth2Session constructor. This ensures requests-oauthlib includes it
    # in the authorization-code token request.
    flow.fetch_token(
        code=code,
        code_verifier=state_payload["code_verifier"],
    )
    return flow.credentials



def _build_service(credentials):
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise ValueError("Gmail authorization expired. Reconnect this account.")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _mailbox_email(credentials) -> str:
    profile = _build_service(credentials).users().getProfile(userId="me").execute()
    email = str(profile.get("emailAddress", "")).strip().lower()
    if not email:
        raise ValueError("Google did not return the mailbox email address.")
    return email


def complete_gmail_authorization(code: str, returned_state: str, supabase, user_id: str | None = None) -> str:
    state_user_id = get_gmail_oauth_user_id(returned_state)
    if user_id is not None and str(user_id) != state_user_id:
        raise ValueError("The Google authorization request belongs to a different app account.")
    user_id = state_user_id
    credentials = _credentials_from_code(code, returned_state)
    mailbox = _mailbox_email(credentials)
    encrypted = _fernet().encrypt(credentials.to_json().encode("utf-8")).decode("ascii")
    supabase.table(CONNECTIONS_TABLE).upsert({
        "user_id": user_id,
        "gmail_account_email": mailbox,
        "encrypted_credentials": encrypted,
        "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }, on_conflict="user_id,gmail_account_email").execute()
    st.session_state["active_gmail_account"] = mailbox
    return mailbox


def list_gmail_connections(supabase, user_id: str) -> list[str]:
    response = (supabase.table(CONNECTIONS_TABLE)
                .select("gmail_account_email")
                .eq("user_id", user_id)
                .order("gmail_account_email")
                .execute())
    return [str(row["gmail_account_email"]).lower() for row in (response.data or [])]


def disconnect_gmail_account(supabase, user_id: str, email: str) -> None:
    supabase.table(CONNECTIONS_TABLE).delete().eq("user_id", user_id).eq(
        "gmail_account_email", email.lower()).execute()
    if st.session_state.get("active_gmail_account") == email.lower():
        st.session_state.pop("active_gmail_account", None)


def get_gmail_credentials(supabase, user_id: str, email: str):
    response = (supabase.table(CONNECTIONS_TABLE)
                .select("encrypted_credentials")
                .eq("user_id", user_id)
                .eq("gmail_account_email", email.lower())
                .limit(1).execute())
    if not response.data:
        raise ValueError("This Gmail account is not connected. Please add it again.")
    encrypted = response.data[0]["encrypted_credentials"]
    try:
        raw = _fernet().decrypt(encrypted.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise ValueError("Could not decrypt this Gmail connection. Check the encryption secret.") from exc
    credentials = Credentials.from_authorized_user_info(json.loads(raw), GMAIL_SCOPES)
    if not credentials.valid and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        new_encrypted = _fernet().encrypt(credentials.to_json().encode()).decode("ascii")
        supabase.table(CONNECTIONS_TABLE).update({
            "encrypted_credentials": new_encrypted,
            "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }).eq("user_id", user_id).eq("gmail_account_email", email.lower()).execute()
    return credentials


def get_gmail_service(account_email: str | None = None, supabase=None, user_id: str | None = None):
    if account_email and supabase is not None and user_id:
        return _build_service(get_gmail_credentials(supabase, user_id, account_email))
    # Compatibility for a previously connected, session-only mailbox.
    credentials = st.session_state.get("gmail_credentials")
    return _build_service(credentials) if credentials else None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}: self.skip += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self.skip: self.skip -= 1
        elif tag.lower() in {"p", "div", "li", "tr"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)
    def text(self):
        return "\n".join(x.strip() for x in "".join(self.parts).splitlines() if x.strip())


def _decode_part(part) -> str:
    try:
        value = part.get_content()
        return value if isinstance(value, str) else str(value)
    except (LookupError, UnicodeDecodeError, TypeError):
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try: return payload.decode(charset, errors="replace")
        except LookupError: return payload.decode("utf-8", errors="replace")


def decode_email_body(message) -> str:
    plain, html = [], []
    def walk(part):
        if part.is_multipart():
            for child in part.iter_parts(): walk(child)
            return
        if part.get_content_disposition() == "attachment": return
        typ = part.get_content_type().lower()
        if typ == "text/plain":
            text = _decode_part(part).strip()
            if text: plain.append(text)
        elif typ == "text/html":
            text = _decode_part(part).strip()
            if text: html.append(text)
    walk(message)
    if plain: return "\n\n".join(plain)
    if html:
        parser = _HTMLTextExtractor()
        for item in html: parser.feed(item)
        parser.close(); return parser.text()
    return ""


def _decode_raw(raw: str) -> bytes:
    if not raw: raise ValueError("Gmail returned an empty raw message.")
    return base64.urlsafe_b64decode((raw + "=" * (-len(raw) % 4)).encode("ascii"))


def fetch_latest_emails(max_results: int = 20, account_email: str | None = None,
                        supabase=None, user_id: str | None = None) -> list[dict]:
    service = get_gmail_service(account_email, supabase, user_id)
    if service is None: raise ValueError("No Gmail account is connected.")
    max_results = max(1, min(int(max_results), 100))
    listing = service.users().messages().list(userId="me", maxResults=max_results).execute()
    emails = []
    for item in listing.get("messages", []):
        details = service.users().messages().get(userId="me", id=item["id"], format="raw").execute()
        raw = _decode_raw(details.get("raw", ""))
        parsed = BytesParser(policy=policy.default).parsebytes(raw)
        sender_name, sender_address = parseaddr(str(parsed.get("From", "")))
        emails.append({
            "id": item["id"], "sender": sender_address or str(parsed.get("From", "")),
            "sender_name": sender_name, "subject": str(parsed.get("Subject", "No Subject")),
            "date": str(parsed.get("Date", "")), "body": decode_email_body(parsed),
            "sha256": hashlib.sha256(raw).hexdigest(), "hash_algorithm": "SHA-256",
            "hash_scope": "Original Gmail RFC822 message bytes", "size_bytes": len(raw),
        })
    return emails
