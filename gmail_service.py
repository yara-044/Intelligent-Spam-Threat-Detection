
import streamlit as st
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
    flow = create_gmail_flow()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )

    st.session_state["gmail_oauth_state"] = state
    return authorization_url


def complete_gmail_authorization(code, returned_state):
    saved_state = st.session_state.get("gmail_oauth_state")

    if not saved_state or returned_state != saved_state:
        raise ValueError("Invalid OAuth state. Please try connecting again.")

    flow = create_gmail_flow(state=saved_state)
    flow.fetch_token(code=code)

    credentials = flow.credentials
    st.session_state["gmail_credentials"] = credentials
    st.session_state.pop("gmail_oauth_state", None)

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
