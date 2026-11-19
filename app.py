from __future__ import annotations
from supabase import create_client
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_and_evaluate import load_data

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
if "supabase_client" not in st.session_state:
    try:
        st.session_state.supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )
    except Exception:
        st.error("Authentication setup error. Please check Supabase Secrets.")
        st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🛡️ Intelligent Spam Detector")
    st.subheader("Welcome")

    login_tab, signup_tab = st.tabs(["Login", "Create Account"])

    with login_tab:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

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
                    st.error("Please verify your email, then try logging in.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with signup_tab:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", use_container_width=True):
            if not new_email or not new_password:
                st.warning("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    response = st.session_state.supabase_client.auth.sign_up({
                        "email": new_email,
                        "password": new_password
                    })
                    st.success("Account created! Check your email if verification is required.")
                except Exception:
                    st.error("Could not create account. Please check your details.")

    st.stop()

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
        "Scan History",
        "Model Dashboard",
        "Dataset Explorer",
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
    st.error(f"Could not load the trained project files: {exc}")
    st.info("Use 'Retrain model' from the sidebar to rebuild the model output.")
    st.stop()

best_name = str(bundle["name"])
model = bundle["model"]

if page == "Message Scanner":
    left, right = st.columns([1.35, .85], gap="large")
    with left:
        st.markdown("### 🔍 Message Scanner")
        st.markdown("<div class='muted'>Paste an email, SMS, or message. The saved best model from your original training script will classify it.</div>", unsafe_allow_html=True)
        sample = st.selectbox("Quick examples", ["Choose an example...", "Normal business email (HAM)", "Suspicious prize message (SPAM)"])
        sample_text = ""
        if sample == "Normal business email (HAM)":
            sample_text = "Subject: Meeting schedule\nHello team, please find the updated meeting agenda attached. Let me know if you have any questions. Best regards."
        elif sample == "Suspicious prize message (SPAM)":
            sample_text = "CONGRATULATIONS! You have won a FREE prize. Click here now to claim your reward. Send your details immediately."
        message = st.text_area("MESSAGE CONTENT", value=sample_text, height=230, placeholder="Paste message text here...")
        if st.button("⚡ ANALYZE MESSAGE", type="primary", use_container_width=True):
           if not message.strip():
                st.warning("Please enter a message first.")
           else:
                try:
                    # Predict using the original trained model
                    prediction = int(model.predict([message])[0])
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
