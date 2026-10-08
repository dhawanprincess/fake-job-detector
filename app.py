import streamlit as st
import joblib
import numpy as np
import re
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FakeGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background-color: #f6f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.2rem;
    }

    .main-title span {
        color: #4f46e5;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Small top badge */
    .badge {
        display: inline-block;
        background-color: #eef2ff;
        color: #4338ca;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    /* Text area */
    textarea {
        border-radius: 14px !important;
        background-color: #ffffff !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 45px !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="badge">AI-POWERED JOB FRAUD DETECTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🛡️ Fake<span>Guard</span> AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Detect potentially fraudulent job postings using machine learning.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "fake_job_detector.joblib"

if not Path(MODEL_PATH).exists():

    st.error(
        "Model not found. Please run `python train_model.py` first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# MAIN AREA
# =========================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large"
)


# =========================================================
# LEFT — JOB INPUT
# =========================================================

with left:

    with st.container(border=True):

        st.subheader("📄 Job Description")

        st.caption(
            "Paste the complete job posting below."
        )

        job_text = st.text_area(
            "Job description",
            height=320,
            placeholder=(
                "Example:\n\n"
                "We are looking for a Software Engineer "
                "with 2+ years of experience..."
            ),
            label_visibility="collapsed"
        )

        if job_text:

            words = len(job_text.split())
            characters = len(job_text)

            st.caption(
                f"📝 {words} words  •  {characters} characters"
            )

        analyze = st.button(
            "🔍  Analyze Job Posting",
            use_container_width=True,
            type="primary"
        )


# =========================================================
# RIGHT — ANALYSIS
# =========================================================

with right:

    with st.container(border=True):

        st.subheader("📊 Risk Assessment")

        st.caption(
            "Your AI-powered analysis will appear here."
        )

        if not analyze:

            st.markdown("")

            st.info(
                "🛡️ Ready to analyze\n\n"
                "Paste a job description on the left "
                "and click **Analyze Job Posting**."
            )

        elif not job_text.strip():

            st.warning(
                "Please paste a job description first."
            )

        else:

            text_proc = preprocess_text(job_text)

            # Dataset labels:
            # 0 = REAL
            # 1 = FAKE

            probabilities = model.predict_proba(
                [text_proc]
            )[0]

            prediction = model.predict(
                [text_proc]
            )[0]

            prob_real = probabilities[0]
            prob_fake = probabilities[1]


            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            if prediction == 1:

                st.error(
                    "🚨 HIGH RISK — POTENTIALLY FRAUDULENT"
                )

                st.metric(
                    "Fraud Probability",
                    f"{prob_fake * 100:.1f}%"
                )

            else:

                st.success(
                    "✅ LOW RISK — LIKELY LEGITIMATE"
                )

                st.metric(
                    "Legitimate Probability",
                    f"{prob_real * 100:.1f}%"
                )


            # ---------------------------------------------
            # PROBABILITY
            # ---------------------------------------------

            st.write("### Confidence Breakdown")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Real",
                    f"{prob_real * 100:.1f}%"
                )

            with col2:

                st.metric(
                    "Fake / Scam",
                    f"{prob_fake * 100:.1f}%"
                )


            st.write("Fake probability")

            st.progress(
                float(prob_fake)
            )


# =========================================================
# SIGNAL ANALYSIS
# =========================================================

if analyze and job_text.strip():

    st.divider()

    st.subheader(
        "🔎 Why did the model make this prediction?"
    )

    try:

        vec = model.named_steps["tfidf"]

        clf = model.named_steps["clf"]

        X_vec = vec.transform(
            [text_proc]
        )

        feature_names = np.array(
            vec.get_feature_names_out()
        )

        coefs = clf.coef_[0]

        contributions = (
            X_vec.toarray()[0] * coefs
        )

        top_fake_idx = np.argsort(
            contributions
        )[-8:][::-1]

        top_real_idx = np.argsort(
            contributions
        )[:8]


        signal_left, signal_right = st.columns(
            2,
            gap="large"
        )


        # ---------------------------------------------
        # FAKE SIGNALS
        # ---------------------------------------------

        with signal_left:

            with st.container(border=True):

                st.markdown(
                    "### 🚨 Signals toward Fake"
                )

                found_fake = False

                for idx in top_fake_idx:

                    if contributions[idx] > 0:

                        found_fake = True

                        st.write(
                            f"**⚠️ {feature_names[idx]}**"
                        )

                        st.caption(
                            f"Fake contribution: "
                            f"+{contributions[idx]:.3f}"
                        )

                        st.divider()

                if not found_fake:

                    st.info(
                        "No strong fake signals detected."
                    )


        # ---------------------------------------------
        # REAL SIGNALS
        # ---------------------------------------------

        with signal_right:

            with st.container(border=True):

                st.markdown(
                    "### ✅ Signals toward Real"
                )

                found_real = False

                for idx in top_real_idx:

                    if contributions[idx] < 0:

                        found_real = True

                        st.write(
                            f"**✓ {feature_names[idx]}**"
                        )

                        st.caption(
                            f"Real contribution: "
                            f"{contributions[idx]:.3f}"
                        )

                        st.divider()

                if not found_real:

                    st.info(
                        "No strong real signals detected."
                    )


    except Exception:

        st.info(
            "Feature explanation is not available "
            "for this prediction."
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("⚙️ Model Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Training Data",
        "17.8K"
    )

with c2:

    st.metric(
        "Text Features",
        "TF-IDF"
    )

with c3:

    st.metric(
        "Classifier",
        "Logistic Regression"
    )

with c4:

    st.metric(
        "Test Accuracy",
        "99%"
    )


# =========================================================
# EXAMPLES
# =========================================================

st.divider()

st.subheader("🧪 Try Example Job Postings")

example1, example2 = st.columns(2)

with example1:

    with st.container(border=True):

        st.markdown(
            "### 🚨 Suspicious Example"
        )

        st.write(
            "Earn $5,000 per week working from home! "
            "No experience required. Pay a small "
            "registration fee of $99 to receive your "
            "starter kit. Limited positions available. "
            "Apply immediately!"
        )


with example2:

    with st.container(border=True):

        st.markdown(
            "### ✅ Legitimate Example"
        )

        st.write(
            "We are looking for a Software Engineer "
            "with 2+ years of experience in Python, "
            "REST APIs and AWS. The selected candidate "
            "will receive competitive salary, medical "
            "insurance and other company benefits."
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ FakeGuard AI is an ML-based screening tool. "
    "A prediction does not prove that a job posting "
    "is fraudulent or legitimate."
)

st.markdown(
    '<div class="footer">'
    '🛡️ FakeGuard AI • Python + Streamlit • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)