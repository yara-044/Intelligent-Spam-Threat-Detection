
import streamlit as st
import secrets
import hashlib
import hmac
from urllib.parse import urlencode
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


def create_gmail_flow(state=None):
    gmail_config = st.secrets["gmail"]

    client_config = {
        "web": {
            "client_id": gmail_config["client_id"],
            "client_secret": gmail_config["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [gmail_config["redirect_uri"]]
        }
    }

    flow = Flow.from_client_config(
        client_config,
        scopes=GMAIL_SCOPES,
        state=state
    )

    flow.redirect_uri = gmail_config["redirect_uri"]
    return flow



def get_gmail_authorization_url():
    import secrets
    import hashlib
    import hmac

    secret = st.secrets["GMAIL_OAUTH_STATE_SECRET"]
    nonce = secrets.token_urlsafe(32)

    signature = hmac.new(
        secret.encode(),
        nonce.encode(),
        hashlib.sha256
    ).hexdigest()

    signed_state = f"{nonce}.{signature}"

    flow = create_gmail_flow(state=signed_state)

    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )

    return authorization_url


def complete_gmail_authorization(code, returned_state):
    import hashlib
    import hmac

    if not returned_state or "." not in returned_state:
        raise ValueError("Invalid OAuth state.")

    nonce, received_signature = returned_state.rsplit(".", 1)

    secret = st.secrets["GMAIL_OAUTH_STATE_SECRET"]

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

    flow = create_gmail_flow(state=returned_state)
    flow.fetch_token(code=code)

    credentials = flow.credentials
    st.session_state["gmail_credentials"] = credentials

    return credentials


def get_gmail_service():
    credentials = st.session_state.get("gmail_credentials")

    if not credentials or not credentials.valid:
        return None

    return build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False
    )
