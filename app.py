from __future__ import annotations
from supabase import create_client
import subprocess
import sys
from pathlib import Path
import joblib
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from train_and_evaluate import load_data
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from train_and_evaluate import make_pipeline 

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "spam_ham_dataset.csv"
TRAIN_SCRIPT = ROOT / "train_and_evaluate.py"
PREDICT_SCRIPT = ROOT / "predict_message.py"
OUTPUT_DIR = ROOT / "model_output"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"

st.set_page_config(
    page_title="Intelligent Spam Detector",
    page_icon="project_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- User Authentication ---------------------------

# --------------------------- User Authentication ---------------------------

if "supabase_client" not in st.session_state:
    try:
        st.session_state.supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
    except Exception:
        st.error("Authentication setup error. Please check Supabase Secrets.")
        st.stop()





# Gmail OAuth Callback
if "code" in st.query_params:
    try:
        from gmail_service import complete_gmail_authorization

        code = st.query_params.get("code")
        returned_state = st.query_params.get("state")

        complete_gmail_authorization(code, returned_state)

        st.query_params.clear()
        st.success("Gmail connected successfully!")

    except Exception as e:
        st.query_params.clear()
        st.error(f"Gmail authorization failed: {e}")




# --------------------------- Authentication State ---------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# --------------------------- Login and Signup ---------------------------

if not st.session_state.authenticated:

    st.title("🛡️ Intelligent Spam Detector")
    st.subheader("Welcome")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:

        email = st.text_input("Email", key="login_email")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):

            try:
                response = st.session_state.supabase_client.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                if response.session:
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()

                else:
                    st.error(
                        "Please verify your email, then try logging in."
                    )

            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:

        new_email = st.text_input("Email", key="signup_email")

        new_password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password"
        )

        if st.button("Create Account", use_container_width=True):

            if not new_email or not new_password:
                st.warning("Please fill in all fields.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

            elif len(new_password) < 8:
                st.error("Passwords must be at least 8 characters.")

            else:
                try:
                    response = st.session_state.supabase_client.auth.sign_up({
                        "email": new_email,
                        "password": new_password
                    })

                    st.success(
                        "Account created! Check your email if verification is required."
                    )

                except Exception:
                    st.error(
                        "Could not create account. Please check your details."
                    )

    st.stop()



def get_current_user():
    response = st.session_state.supabase_client.auth.get_user()
    return response.user


def save_custom_model(model_bundle, user_id):
    model_path = f"{user_id}/custom_spam_model.joblib"

    model_buffer = io.BytesIO()
    joblib.dump(model_bundle, model_buffer)

    st.session_state.supabase_client.storage.from_(
        "custom-models"
    ).upload(
        model_path,
        model_buffer.getvalue(),
        file_options={
            "upsert": "true",
            "content-type": "application/octet-stream"
        }
    )


def load_custom_model(user_id):
    model_path = f"{user_id}/custom_spam_model.joblib"

    file_data = st.session_state.supabase_client.storage.from_(
        "custom-models"
    ).download(model_path)

    return joblib.load(io.BytesIO(file_data))


try:
    current_user = get_current_user()

    if current_user:
        loaded_user_id = st.session_state.get("custom_model_loaded_user")

        if loaded_user_id != current_user.id:
            try:
                saved_bundle = load_custom_model(current_user.id)

                st.session_state["custom_trained_models"] = {
                    saved_bundle["name"]: saved_bundle["model"]
                }

                st.session_state["custom_best_model"] = saved_bundle["name"]
                st.session_state["custom_model_details"] = saved_bundle
                st.session_state["custom_model_loaded_user"] = current_user.id

            except Exception:
                st.session_state.pop("custom_trained_models", None)
                st.session_state.pop("custom_best_model", None)
                st.session_state["custom_model_loaded_user"] = current_user.id

except Exception as e:
    st.warning(f"Could not restore your custom model: {e}")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root { --gold:#f1e900; --gold2:#c9c300; --muted:#aaa99a; --line:#343323; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background:radial-gradient(ellipse at 50% -20%,#24230a 0%,#090909 42%,#030303 100%); color:#f4f2d9; }
[data-testid="stSidebar"] { background:#080808; border-right:1px solid #34320b; }
.block-container { padding-top:1.3rem; max-width:1500px; }
h1,h2,h3 { color:#fffef0 !important; letter-spacing:-.025em; }
p,label,.stMarkdown { color:#e4e1c8; }
.hero { padding:24px 28px; border:1px solid #5b5700; border-radius:18px;
background:linear-gradient(120deg,rgba(26,25,5,.98),rgba(9,9,8,.98));
box-shadow:0 16px 45px rgba(0,0,0,.4); margin-bottom:20px; }
.eyebrow { color:var(--gold); text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:.2em; }
.hero-title { font-size:clamp(25px,3vw,38px); font-weight:900; line-height:1.12; color:#fff; margin:8px 0; }
.hero-sub,.muted { color:#b8b59b; font-size:13px; }
.panel { background:rgba(17,17,13,.96); border:1px solid #393719; border-radius:15px; padding:18px 20px; }
.metric { background:linear-gradient(145deg,#1b1a0a,#10100c); border:1px solid #45420f; border-radius:14px; padding:16px; }
.metric-label { color:#c7c39a; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#fffde0; font-size:26px; font-weight:800; margin-top:5px; }
.result-safe { border:1px solid rgba(241,233,0,.55); background:rgba(241,233,0,.07); border-radius:15px; padding:20px; }
.result-spam { border:1px solid rgba(255,89,72,.65); background:rgba(255,89,72,.08); border-radius:15px; padding:20px; }
.result-title { font-size:25px; font-weight:800; margin:7px 0; }
.stTextArea textarea, .stTextInput input { background:#0c0c09 !important; color:#fffde8 !important; border:1px solid #4a471a !important; border-radius:10px !important; }
.stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div { background:#0c0c09; border-color:#4a471a; }
div.stButton > button[kind="primary"], div.stDownloadButton > button { background:linear-gradient(90deg,#d5ce00,#f1e900); color:#080808; border:0; border-radius:9px; font-weight:900; min-height:44px; }
div.stButton > button { border:1px solid #57520a; border-radius:9px; color:#f3f0c9; background:#15150d; }
div.stButton > button:hover { border-color:#f1e900; color:#fff; }
[data-testid="stMetric"] { background:#15150d; padding:13px; border-radius:11px; border:1px solid #45420f; }
[data-testid="stDataFrame"] { border:1px solid #393719; border-radius:10px; }
hr { border-color:#38350c; }
footer {visibility:hidden;}
.logo-wrap { display:flex; justify-content:center; padding:0 0 10px 0; }
.logo-wrap img { width:100%; max-width:235px; border-radius:8px; }
.sidebar-brand { color:#f1e900; font-weight:900; letter-spacing:.12em; font-size:13px; text-align:center; }
div[data-testid="stRadio"] label { color:#e8e4c5; }
</style>
""", unsafe_allow_html=True)

def model_ready() -> bool:
    return MODEL_PATH.exists() and METRICS_PATH.exists()

def run_training():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data", str(DATA_PATH),
        "--output", str(OUTPUT_DIR),
    ]
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    return result

def load_bundle():
    return joblib.load(MODEL_PATH)

def load_metrics():
    return pd.read_csv(METRICS_PATH)

def render_metric(label, value):
    st.markdown(
        f"<div class='metric'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### 👤 Account")

    if st.button("Logout", use_container_width=True):
        st.session_state.supabase_client.auth.sign_out()
        st.session_state.authenticated = False
        st.rerun()

    st.markdown("---")
    st.image(str(ROOT / "project_logo.png"), use_container_width=True)
    st.markdown("<div class='sidebar-brand'>INTELLIGENT SECURITY CONSOLE</div>", unsafe_allow_html=True)
    st.caption("SPAM & THREAT ANALYSIS")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Message Scanner",
            "Email Security",
            "Scan History",
            "Model Dashboard",
            "Dataset Explorer",
            "Custom Dataset Training",
            "Saved Model Details",
            "About Project"
        ]
    )
st.markdown("""
<div class="hero">
  <div class="eyebrow">Intelligent message security</div>
  <div class="hero-title">Intelligent Spam & Threat Detection System</div>
  <div class="hero-sub">Analyze message content, detect spam patterns, and review model performance through one security console.</div>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("spam_ham_dataset.csv was not found in the application folder.")
    st.stop()

if not model_ready():
    st.warning("The original model has not been trained yet. Train all four classifiers to create best_model.joblib and metrics.csv.")
    if st.button("🚀 TRAIN PROJECT MODELS", type="primary", use_container_width=True):
        with st.spinner("Training Naive Bayes, Logistic Regression, Linear SVM and Random Forest..."):
            result = run_training()
        if result.returncode == 0:
            st.success("Training finished successfully.")
            st.code(result.stdout)
            st.rerun()
        else:
            st.error("The training script returned an error.")
            st.code(result.stderr or result.stdout)
    st.stop()


try:
    bundle = load_bundle()
    metrics_df = load_metrics()
    dataset = load_data(DATA_PATH)
   

   

except Exception as exc:
    st.error("Project loading failed:")
    st.exception(exc)
    st.stop()

best_name = str(bundle["name"])
model = bundle["model"]


if page == "Message Scanner":
    left, right = st.columns([1.35, .85], gap="large")
    with left:
        st.markdown("### 🔍 Message Scanner")
        st.markdown("<div class='muted'>Paste an email, SMS, or message. The saved best model from your original training script will classify it.</div>", unsafe_allow_html=True)
        
        if "quick_examples" not in st.session_state:
            ham_examples = dataset[
                dataset["label"].astype(str).str.lower() == "ham"
            ]
            spam_examples = dataset[
                dataset["label"].astype(str).str.lower() == "spam"
            ]

            parts = []

            if not ham_examples.empty:
                parts.append(
                    ham_examples.sample(n=min(5, len(ham_examples)))
                )

            if not spam_examples.empty:
                parts.append(
                    spam_examples.sample(n=min(5, len(spam_examples)))
                )

            if parts:
                examples = pd.concat(parts).sample(frac=1).reset_index(drop=True)

                st.session_state["quick_examples"] = [
                    {
                        "display": f"{row['label'].upper()} #{i+1} — {str(row['text'])[:75]}",
                        "text": str(row["text"])
                    }
                    for i, row in examples.iterrows()
                ]
            else:
                st.session_state["quick_examples"] = []

        quick_examples = st.session_state["quick_examples"]

        sample_options = ["Choose an example..."] + [
            item["display"] for item in quick_examples
        ]

        
        if "message_input" not in st.session_state:
            st.session_state["message_input"] = ""

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⚠️ Generate SPAM", use_container_width=True):
                examples = dataset[
                    dataset["label"].astype(str).str.lower() == "spam"
                ]["text"]

                if not examples.empty:
                    st.session_state["message_input"] = examples.sample(1).iloc[0]
                    st.rerun()

        with col2:
            if st.button("✅ Generate HAM", use_container_width=True):
                examples = dataset[
                    dataset["label"].astype(str).str.lower() == "ham"
                ]["text"]

                if not examples.empty:
                    st.session_state["message_input"] = examples.sample(1).iloc[0]
                    st.rerun()

        with col3:
            if st.button("✍️ Write My Own", use_container_width=True):
                st.session_state["message_input"] = ""
                st.rerun()

        message = st.text_area(
            "MESSAGE CONTENT",
            height=230,
            placeholder="Generate an example or write your own message...",
            key="message_input"
        )

        model_options = ["Original Project Model"]

        if "custom_trained_models" in st.session_state:
            model_options.append("Custom Trained Model")

        selected_model = st.selectbox(
            "🤖 Select Detection Model",
            model_options
        )

        active_model = model

        if selected_model == "Custom Trained Model":
            active_model = st.session_state["custom_trained_models"][
                st.session_state["custom_best_model"]
            ]

        if st.button("⚡ ANALYZE MESSAGE", type="primary", use_container_width=True):
           if not message.strip():
                st.warning("Please enter a message first.")
           else:
                try:
                    # Predict using the original trained model
                    prediction = int(active_model.predict([message])[0])
                    label = "spam" if prediction == 1 else "ham"

                    # Display the analysis result
                    st.session_state["scan_result"] = {
                        "message": message,
                        "label": label
                    }

                    # Save scan to Supabase history
                    user_response = st.session_state.supabase_client.auth.get_user()
                    current_user = user_response.user

                    if current_user:
                        st.session_state.supabase_client.table("scan_history").insert({
                            "user_id": current_user.id,
                            "message": message,
                            "prediction": label
                        }).execute()

                        st.success("Analysis completed and saved to your history.")
                    else:
                        st.warning("Analysis completed, but the scan could not be saved.")

                except Exception as e:
                    st.error(f"Analysis or history saving failed: {e}")
    with right:
        st.markdown("### 📡 Engine Status")
        st.markdown(f"<div class='panel'>🟢 <b>Model loaded</b><br><span class='muted'>Saved model bundle is active</span><hr><span class='muted'>Selected classifier</span><br><b>{best_name}</b><br><br><span class='muted'>Dataset records after duplicate removal</span><br><b>{len(dataset):,}</b><br><br><span class='muted'>Feature extraction</span><br><b>TF-IDF · 1–2 grams</b></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 Detection pipeline")
        st.markdown("<div class='panel'><b>01</b> &nbsp; Input message<br>↓<br><b>02</b> &nbsp; TF-IDF transformation<br>↓<br><b>03</b> &nbsp; Saved best classifier<br>↓<br><b>04</b> &nbsp; HAM / SPAM result</div>", unsafe_allow_html=True)
    result = st.session_state.get("scan_result")
    if result:
        st.markdown("---")
        st.markdown("### 🧾 Analysis Result")
        if result["label"] == "spam":
            st.markdown("<div class='result-spam'><div class='eyebrow' style='color:#ff8497'>⚠ THREAT ALERT</div><div class='result-title' style='color:#ff8497'>SPAM / SUSPICIOUS MESSAGE</div><div class='muted'>The original trained model classified this message as SPAM.</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-safe'><div class='eyebrow'>✓ MESSAGE CLEARED</div><div class='result-title' style='color:#38d9c4'>HAM / LEGITIMATE MESSAGE</div><div class='muted'>The original trained model classified this message as HAM.</div></div>", unsafe_allow_html=True)
        with st.expander("View analyzed message"):
            st.code(result["message"], language=None)
        st.caption("The prediction follows predict_message.py. The current prediction script returns a class label only, so no confidence percentage is displayed.")


elif page == "Email Security":
    st.markdown("### 📧 Email Security")
    st.write("Connect your Gmail account to scan incoming emails.")

    if "gmail_credentials" not in st.session_state:
        try:
            from gmail_service import get_gmail_authorization_url

            st.info("Connect your Gmail account to start email security scanning.")

            if st.button("🔗 Connect Gmail", type="primary"):
                auth_url = get_gmail_authorization_url()
                st.link_button(
                    "Authorize Gmail Account",
                    auth_url,
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Gmail connection error: {e}")

    else:
        st.success("Gmail account connected successfully!")

elif page == "Scan History":
    st.markdown("### 🕘 Scan History")
    st.caption("Your previous message analysis results.")

    try:
        user_response = st.session_state.supabase_client.auth.get_user()
        current_user = user_response.user

        if current_user:
            history_response = (
                st.session_state.supabase_client
                .table("scan_history")
                .select("id, message, prediction, created_at")
                .eq("user_id", current_user.id)
                .order("created_at", desc=True)
                .execute()
            )

            history = history_response.data

            if history:
                st.markdown(f"**Total scans: {len(history)}**")

                for item in history:
                    label = item["prediction"].upper()
                    icon = "⚠️" if label == "SPAM" else "✅"

                    with st.expander(
                        f"{icon} {label} — {item['created_at']}"
                    ):
                        st.write("**Analyzed message:**")
                        st.code(item["message"], language=None)

                        if st.button(
                            "🗑️ Delete This Record",
                            key=f"delete_{item['id']}"
                        ):
                            (
                                st.session_state.supabase_client
                                .table("scan_history")
                                .delete()
                                .eq("id", item["id"])
                                .eq("user_id", current_user.id)
                                .execute()
                            )

                            st.success("Record deleted successfully.")
                            st.rerun()

            else:
                st.info("You haven't analyzed any messages yet.")

    except Exception as e:
        st.error(f"Could not load scan history: {e}")

elif page == "Model Dashboard":
    st.markdown("### 📊 Model Performance Dashboard")
    st.caption("These are the values generated by your original train_and_evaluate.py run and saved in model_output/metrics.csv.")
    best_row = metrics_df.loc[metrics_df["model"] == best_name].iloc[0]
    cols = st.columns(5)
    for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                               ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
        with col:
            render_metric(label, f"{float(best_row[key])*100:.2f}%")
    st.markdown("#### Classifier comparison")
    show = metrics_df.melt(id_vars="model", value_vars=["accuracy","precision","recall","f1","roc_auc"],
                           var_name="Metric", value_name="Score")
    fig = px.bar(show, x="model", y="Score", color="Metric", barmode="group",
                 range_y=[0,1], template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title="", xaxis_title="", yaxis_title="Score",
                      margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Full evaluation metrics")
    st.dataframe(metrics_df.style.format({c:"{:.4f}" for c in metrics_df.columns if c != "model"}),
                 use_container_width=True, hide_index=True)
    st.info(f"Best model selected by spam F1-score in your training script: {best_name}.")
    for image_name, caption in [("model_comparison.png","Model comparison chart"),("confusion_matrices.png","Confusion matrices")]:
        image_path = OUTPUT_DIR / image_name
        if image_path.exists():
            st.markdown(f"#### {caption}")
            st.image(str(image_path), use_container_width=True)

elif page == "Dataset Explorer":
    st.markdown("### 🗂️ Dataset Explorer")
    ham_count = int((dataset["label_num"] == 0).sum())
    spam_count = int((dataset["label_num"] == 1).sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("Validated unique messages", f"{len(dataset):,}")
    c2.metric("HAM", f"{ham_count:,}")
    c3.metric("SPAM", f"{spam_count:,}")
    st.caption("Dataset validation, label consistency checks and duplicate removal follow load_data() in train_and_evaluate.py.")
    query = st.text_input("Search message text")
    labels = st.multiselect("Filter class", ["ham","spam"], default=["ham","spam"])
    view = dataset[dataset["label"].isin(labels)].copy()
    if query.strip():
        view = view[view["text"].str.contains(query, case=False, na=False)]
    st.dataframe(view[["label","label_num","text"]], use_container_width=True, height=430)
    st.download_button("⬇ Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_spam_ham_dataset.csv", mime="text/csv")

elif page == "Custom Dataset Training":
    st.markdown("### 📂 Custom Dataset Training")
    st.caption("Upload your own CSV dataset and detect its columns automatically.")

    uploaded_file = st.file_uploader(
        "Upload your CSV Dataset",
        type=["csv"],
        key="custom_dataset"
    )

    if uploaded_file is not None:
        try:
             uploaded_file.seek(0)
             custom_data = pd.read_csv(
             uploaded_file,
             encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            custom_data = pd.read_csv(
            uploaded_file,
            encoding="latin-1"
           )
            if custom_data.empty or len(custom_data.columns) < 2:
                st.error("The dataset must contain at least two columns.")
            else:
                st.success("Dataset uploaded successfully!")

                st.markdown("#### Dataset Preview")
                st.dataframe(custom_data.head(10), use_container_width=True)

                st.markdown("#### Dataset Information")
                st.write(f"Rows: {custom_data.shape[0]}")
                st.write(f"Columns: {custom_data.shape[1]}")

                columns = list(custom_data.columns)

                text_keywords = [
                    "text", "message", "content", "body",
                    "email", "sms", "sentence", "comment"
                ]
                label_keywords = [
                    "label", "class", "category", "target",
                    "type", "status", "classification"
                ]

                def find_column(keywords):
                    for col in columns:
                        name = str(col).strip().lower()
                        if any(word in name for word in keywords):
                            return col
                    return None

                detected_text = find_column(text_keywords)
                detected_label = find_column(label_keywords)

                if detected_text is None:
                    string_columns = [
                        col for col in columns
                        if custom_data[col].dtype == "object"
                    ]
                    if string_columns:
                        detected_text = max(
                            string_columns,
                            key=lambda col: custom_data[col].astype(str).str.len().mean()
                        )

                if detected_label is None:
                    possible_labels = [
                        col for col in columns
                        if custom_data[col].nunique(dropna=True) <= 20
                        and custom_data[col].nunique(dropna=True) >= 2
                    ]
                    if possible_labels:
                        detected_label = min(
                            possible_labels,
                            key=lambda col: custom_data[col].nunique(dropna=True)
                        )

                st.markdown("#### 🔎 Automatic Column Detection")

                if detected_text is None or detected_label is None:
                    st.warning("Automatic detection was incomplete. Please select the columns manually.")
                else:
                    st.success("Columns detected! You can change them if needed.")

                text_index = (
                    columns.index(detected_text)
                    if detected_text is not None else 0
                )

                label_index = (
                    columns.index(detected_label)
                    if detected_label is not None else min(1, len(columns) - 1)
                )

                text_column = st.selectbox(
                    "Select the message/text column",
                    columns,
                    index=text_index,
                    key="custom_text_column"
                )

                label_column = st.selectbox(
                    "Select the class/label column",
                    columns,
                    index=label_index,
                    key="custom_label_column"
                )

                if text_column == label_column:
                    st.error("Please select two different columns.")
                else:
                    st.markdown("#### Selected Columns")
                    st.write(f"Message column: **{text_column}**")
                    st.write(f"Label column: **{label_column}**")

                    
                    st.markdown("#### 🏷️ Label Classification")

                    label_values = (
                        custom_data[label_column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    unique_labels = sorted(label_values.unique().tolist())

                    st.write("Detected labels:", unique_labels)

                    if len(unique_labels) != 2:
                        st.warning(
                            "This version supports binary classification only. "
                            "The dataset must contain exactly two different labels."
                        )
                    else:
                        spam_candidates = [
                            "spam", "junk", "phishing", "malicious",
                            "fraud", "scam", "suspicious", "1"
                        ]

                        detected_spam = next(
                            (
                                value for value in unique_labels
                                if value.lower() in spam_candidates
                            ),
                            None
                        )

                        default_index = (
                            unique_labels.index(detected_spam)
                            if detected_spam is not None else 0
                        )

                        spam_label = st.selectbox(
                            "Which label represents SPAM?",
                            unique_labels,
                            index=default_index,
                            key="custom_spam_label"
                        )

                        st.info(
                            "The selected label will become SPAM. "
                            "The other label will become HAM."
                        )

                        prepared_data = pd.DataFrame({
                            "text": custom_data[text_column],
                            "label": custom_data[label_column]
                        })

                        prepared_data = prepared_data.dropna(
                            subset=["text", "label"]
                        ).copy()

                        prepared_data["text"] = (
                            prepared_data["text"].astype(str).str.strip()
                        )

                        prepared_data["label"] = (
                            prepared_data["label"].astype(str).str.strip()
                        )

                        prepared_data = prepared_data[
                            prepared_data["text"] != ""
                        ]

                        prepared_data["label"] = prepared_data["label"].apply(
                            lambda value: (
                                "spam" if value == spam_label else "ham"
                            )
                        )

                        prepared_data = prepared_data.drop_duplicates(
                            subset=["text"]
                        )

                        st.markdown("#### 📊 Prepared Dataset")

                        st.write(
                            prepared_data["label"].value_counts()
                        )

                        st.dataframe(
                            prepared_data.head(10),
                            use_container_width=True
                        )

                        if prepared_data["label"].nunique() == 2:
                            st.success(
                                "Dataset labels successfully converted to HAM and SPAM."
                            )
                        else:
                            st.error(
                                "The dataset must contain examples from both classes."
                            )

                        class_counts = prepared_data["label"].value_counts()

                        if (
                            prepared_data["label"].nunique() == 2
                            and class_counts.min() >= 5
                        ):
                            st.markdown("---")
                            st.markdown("### 🚀 Train & Compare Models")

                            if st.button(
                                "START MODEL TRAINING",
                                type="primary",
                                use_container_width=True
                            ):
                                X = prepared_data["text"]
                                y = prepared_data["label"].map({
                                    "ham": 0,
                                    "spam": 1
                                })

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=0.2,
                                    random_state=42,
                                    stratify=y
                                )

                                classifiers = {
                                    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
                                    "Logistic Regression": LogisticRegression(
                                        max_iter=2000,
                                        class_weight="balanced"
                                    ),
                                    "Linear SVM": LinearSVC(
                                        class_weight="balanced",
                                        random_state=42
                                    ),
                                    "Random Forest": RandomForestClassifier(
                                        n_estimators=300,
                                        class_weight="balanced",
                                        random_state=42,
                                        n_jobs=-1
                                    )
                                }

                                results = []
                                trained_models = {}

                                try:
                                    with st.spinner(
                                        "Training four models. Please wait..."
                                    ):
                                        for name, classifier in classifiers.items():
                                            custom_model = make_pipeline(classifier)
                                            custom_model.fit(X_train, y_train)

                                            predictions = custom_model.predict(X_test)

                                            results.append({
                                                "Model": name,
                                                "Accuracy": accuracy_score(
                                                    y_test, predictions
                                                ),
                                                "Precision": precision_score(
                                                    y_test, predictions,
                                                    zero_division=0
                                                ),
                                                "Recall": recall_score(
                                                    y_test, predictions,
                                                    zero_division=0
                                                ),
                                                "F1-score": f1_score(
                                                    y_test, predictions,
                                                    zero_division=0
                                                )
                                            })

                                            trained_models[name] = custom_model

                                        original_predictions = model.predict(X_test)

                                        results.append({
                                            "Model": "Original Project Model",
                                            "Accuracy": accuracy_score(
                                                y_test, original_predictions
                                            ),
                                            "Precision": precision_score(
                                                y_test, original_predictions,
                                                zero_division=0
                                            ),
                                            "Recall": recall_score(
                                                y_test, original_predictions,
                                                zero_division=0
                                            ),
                                            "F1-score": f1_score(
                                                y_test, original_predictions,
                                                zero_division=0
                                            )
                                        })

                                    results_df = pd.DataFrame(results)
                                    results_df = results_df.sort_values(
                                        "F1-score",
                                        ascending=False
                                    )

                                    
                                    best_custom_name = results_df[
                                        results_df["Model"] != "Original Project Model"
                                    ].iloc[0]["Model"]

                                    st.session_state["custom_training_results"] = results_df
                                    st.session_state["custom_trained_models"] = trained_models
                                    st.session_state["custom_best_model"] = best_custom_name

                                    try:
                                        current_user = get_current_user()

                                        
                                        
                                        if current_user:
                                            best_row = results_df[
                                                results_df["Model"] == best_custom_name
                                            ].iloc[0]

                                            model_details = {
                                                "model": trained_models[best_custom_name],
                                                "name": best_custom_name,
                                                "dataset_name": uploaded_file.name,
                                                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "accuracy": float(best_row["Accuracy"]),
                                                "precision": float(best_row["Precision"]),
                                                "recall": float(best_row["Recall"]),
                                                "f1_score": float(best_row["F1-score"])
                                            }

                                            save_custom_model(
                                                model_details,
                                                current_user.id
                                            )

                                            st.session_state["custom_model_loaded_user"] = current_user.id
                                            st.session_state["custom_model_details"] = model_details

                                            st.success("Your custom model was saved securely to your account.")

                                    except Exception as e:
                                        st.warning(f"Model trained, but could not be saved: {e}")

                                    st.success("Training completed successfully!")

                                except Exception as e:
                                    st.error(f"Training failed: {e}")


                        elif prepared_data["label"].nunique() == 2:
                            st.warning(
                                "Each class needs at least 5 valid messages for training."
                            )

                        if "custom_training_results" in st.session_state:
                            st.markdown("### 📊 Model Comparison")

                            st.dataframe(
                                st.session_state["custom_training_results"].style.format({
                                    "Accuracy": "{:.2%}",
                                    "Precision": "{:.2%}",
                                    "Recall": "{:.2%}",
                                    "F1-score": "{:.2%}"
                                }),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.success(
                                "Best custom model: "
                                + st.session_state["custom_best_model"]
                            )

                            st.markdown("#### 💾 Save Your Custom Model")

                            best_model_name = st.session_state["custom_best_model"]

                            model_buffer = io.BytesIO()

                            joblib.dump(
                                {
                                    "model": st.session_state["custom_trained_models"][best_model_name],
                                    "name": best_model_name
                                },
                                model_buffer
                            )

                            st.download_button(
                                "⬇️ Download Best Custom Model",
                                data=model_buffer.getvalue(),
                                file_name="custom_spam_model.joblib",
                                mime="application/octet-stream",
                                use_container_width=True
                            )


        except Exception as e:
            st.error(f"Could not read dataset: {e}")



elif page == "Saved Model Details":
    st.markdown("### 💾 Saved Custom Model")
    st.caption("Information about your saved custom detection model.")

    details = st.session_state.get("custom_model_details")

    if details:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Name", details.get("name", "Unknown"))

        with col2:
            st.metric("Dataset", details.get("dataset_name", "Unknown"))

        with col3:
            st.metric("Training Date", details.get("trained_at", "Unknown"))

        st.markdown("### 📊 Model Performance")

        if all(key in details for key in ["accuracy", "precision", "recall", "f1_score"]):
            metric_cols = st.columns(4)

            metric_cols[0].metric("Accuracy", f"{details['accuracy']:.2%}")
            metric_cols[1].metric("Precision", f"{details['precision']:.2%}")
            metric_cols[2].metric("Recall", f"{details['recall']:.2%}")
            metric_cols[3].metric("F1-Score", f"{details['f1_score']:.2%}")
        else:
            st.info("Performance metrics are not available. Train the model again to save them.")

        st.success("Your custom model is available and loaded.")

    else:
        st.info("No saved custom model details are available yet.")
 
else:
    st.markdown("### 🛡️ About the Project")
    st.markdown("""
    <div class="panel">
    <h3>Intelligent Spam & Threat Detection System</h3>
    This project treats message filtering as a supervised binary classification task.
    <hr>
    <b>HAM (Label 0)</b><br>Legitimate, wanted communication.
    <br><br><b>SPAM (Label 1)</b><br>Unsolicited or suspicious message content.
    <hr>
    <b>Original project pipeline</b><br>
    Dataset validation → duplicate removal → TF-IDF features → four classifiers → model comparison → saved best model → message prediction.
    <hr>
    <b>Classifiers in the supplied training code</b><br>
    Multinomial Naive Bayes · Logistic Regression · Linear SVM · Random Forest.
    </div>
    """, unsafe_allow_html=True)
    st.warning("The dataset contains only HAM and SPAM labels. This application classifies those two classes; it does not independently verify malware, URLs, sender identity, or determine a specific threat subtype.")
    st.markdown("**Project team**")
    st.write("Yara Alamoush · Adnan Al Jarrah · Rashed AbuSara")
    st.write("Supervisor: Ahmad Irjoob")
    st.write("Faculty of Artificial Intelligence — Al-Balqa Applied University")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7187a2;font-size:11px'>INTELLIGENT SPAM & THREAT DETECTION SYSTEM · ORIGINAL PROJECT MODEL</div>", unsafe_allow_html=True)
