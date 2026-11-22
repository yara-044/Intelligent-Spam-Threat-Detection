
import streamlit as st
import secrets
import hashlib
import hmac

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build


# Gmail permissions: read emails only
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


# Create Google OAuth Flow
def create_gmail_flow(state=None, code_verifier=None):

    gmail_config = st.secrets["gmail"]

    client_config = {
        "web": {
            "client_id": gmail_config["client_id"],
            "client_secret": gmail_config["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                gmail_config["redirect_uri"]
            ]
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        state=state,
        code_verifier=code_verifier
    )

    flow.redirect_uri = gmail_config["redirect_uri"]

    return flow


# Generate Gmail Authorization URL
def get_gmail_authorization_url():

    secret = st.secrets["GMAIL_OAUTH_STATE_SECRET"]

    # Generate secure random state
    nonce = secrets.token_urlsafe(32)

    # Sign OAuth state
    signature = hmac.new(
        secret.encode(),
        nonce.encode(),
        hashlib.sha256
    ).hexdigest()

    signed_state = f"{nonce}.{signature}"

    # Generate a reproducible PKCE verifier
    code_verifier = hmac.new(
        secret.encode(),
        f"pkce:{nonce}".encode(),
        hashlib.sha256
    ).hexdigest()

    flow = create_gmail_flow(
        state=signed_state,
        code_verifier=code_verifier
    )

    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )

    return authorization_url


# Complete Gmail OAuth Authorization
def complete_gmail_authorization(code, returned_state):

    if not code:
        raise ValueError("Google authorization code is missing.")

    if not returned_state or "." not in returned_state:
        raise ValueError("Invalid OAuth state.")

    # Separate nonce and signature
    nonce, received_signature = returned_state.rsplit(".", 1)

    secret = st.secrets["GMAIL_OAUTH_STATE_SECRET"]

    # Verify OAuth state signature
    expected_signature = hmac.new(
        secret.encode(),
        nonce.encode(),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(
        received_signature,
        expected_signature
    ):
        raise ValueError("Invalid OAuth state signature.")

    # Recreate the same PKCE verifier
    code_verifier = hmac.new(
        secret.encode(),
        f"pkce:{nonce}".encode(),
        hashlib.sha256
    ).hexdigest()

    # Recreate OAuth flow
    flow = create_gmail_flow(
        state=returned_state,
        code_verifier=code_verifier
    )

    # Exchange authorization code for credentials
    flow.fetch_token(code=code)

    credentials = flow.credentials

    # Save credentials in current Streamlit session
    st.session_state["gmail_credentials"] = credentials

    return credentials


# Get Gmail API Service
def get_gmail_service():

    credentials = st.session_state.get(
        "gmail_credentials"
    )

    if not credentials:
        return None

    # Refresh expired credentials when possible
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
        cache_discovery=False
    )

import base64
from email.utils import parseaddr


def decode_email_body(payload):
    """Extract readable email body from Gmail payload."""

    body = ""

    if "parts" in payload:
        for part in payload["parts"]:
            body += decode_email_body(part)

    elif payload.get("body", {}).get("data"):
        data = payload["body"]["data"]

        decoded = base64.urlsafe_b64decode(
            data + "=" * (-len(data) % 4)
        )

        mime_type = payload.get("mimeType", "")

        if mime_type == "text/plain":
            body = decoded.decode("utf-8", errors="replace")

    return body


def fetch_latest_emails(max_results=20):
    """Fetch the latest Gmail messages."""

    service = get_gmail_service()

    if service is None:
        raise ValueError(
            "Gmail session expired. Please connect Gmail again."
        )

    response = service.users().messages().list(
        userId="me",
        maxResults=max_results
    ).execute()

    messages = response.get("messages", [])
    emails = []

    for message in messages:
        details = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        payload = details.get("payload", {})
        headers = payload.get("headers", [])

        header_map = {
            item["name"].lower(): item["value"]
            for item in headers
        }

        body = decode_email_body(payload)

        emails.append({
            "id": message["id"],
            "sender": parseaddr(
                header_map.get("from", "")
            )[1],
            "subject": header_map.get("subject", "No Subject"),
            "date": header_map.get("date", ""),
            "body": body
        })

    return emails
