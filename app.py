import streamlit as st
import joblib
import numpy as np
import re
from pathlib import Path


# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Posting Detector")
st.write(
    "Paste a job description below and the model will predict "
    "whether it is likely **REAL** or **FAKE / SCAM**."
)


# =========================
# Load trained model
# =========================

MODEL_PATH = "fake_job_detector.joblib"

if not Path(MODEL_PATH).exists():
    st.error(
        "Model not found. Please run `python train_model.py` first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# =========================
# Text preprocessing
# =========================

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# Input
# =========================

job_text = st.text_area(
    "📄 Paste Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)


# =========================
# Prediction
# =========================

if st.button("🔍 Analyze Job", use_container_width=True):

    if not job_text.strip():

        st.warning("Please paste a job description first.")

    else:

        text_proc = preprocess_text(job_text)

        # Dataset labels:
        # 0 = REAL
        # 1 = FAKE

        probabilities = model.predict_proba([text_proc])[0]

        prediction = model.predict([text_proc])[0]

        prob_real = probabilities[0]
        prob_fake = probabilities[1]


        # =========================
        # Result
        # =========================

        st.markdown("---")

        if prediction == 1:

            st.error("🚨 Prediction: FAKE / SCAM")

            st.metric(
                "Fake Probability",
                f"{prob_fake * 100:.1f}%"
            )

        else:

            st.success("✅ Prediction: REAL")

            st.metric(
                "Real Probability",
                f"{prob_real * 100:.1f}%"
            )


        # =========================
        # Probability breakdown
        # =========================

        st.subheader("📊 Model Probability")

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


        st.progress(float(prob_fake))

        st.caption(
            "These are model probabilities, not guaranteed certainty."
        )


        # =========================
        # Feature contribution
        # =========================

        try:

            vec = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]

            X_vec = vec.transform([text_proc])

            feature_names = np.array(
                vec.get_feature_names_out()
            )

            coefs = clf.coef_[0]

            contributions = X_vec.toarray()[0] * coefs

            # For this model:
            # positive contribution → FAKE
            # negative contribution → REAL

            top_fake_idx = np.argsort(contributions)[-8:][::-1]

            top_real_idx = np.argsort(contributions)[:8]

            st.subheader("🔎 Important Signals")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🚨 Fake signals")

                for idx in top_fake_idx:

                    if contributions[idx] > 0:

                        st.write(
                            f"**{feature_names[idx]}** "
                            f"→ +{contributions[idx]:.3f}"
                        )

            with col2:

                st.markdown("### ✅ Real signals")

                for idx in top_real_idx:

                    if contributions[idx] < 0:

                        st.write(
                            f"**{feature_names[idx]}** "
                            f"→ {contributions[idx]:.3f}"
                        )

        except Exception as e:

            st.info(
                "Feature explanation is not available for this prediction."
            )


# =========================
# Example jobs
# =========================

st.markdown("---")

st.subheader("🧪 Try an Example")

col1, col2 = st.columns(2)

with col1:

    if st.button("🚨 Fake Example"):

        st.code(
            """
Earn $5,000 per week working from home!
No experience required.

Pay a small registration fee of $99
to receive your starter kit.

Limited positions available.
Apply immediately!
            """
        )


with col2:

    if st.button("✅ Real Example"):

        st.code(
            """
We are looking for a Software Engineer
with 2+ years of experience in Python,
REST APIs and AWS.

The selected candidate will receive
competitive salary, medical insurance,
and other company benefits.

Apply through our official careers portal.
            """
        )


st.markdown("---")

st.caption(
    "⚠️ This system provides an ML-based prediction and should "
    "not be treated as definitive proof that a job is fraudulent."
)
