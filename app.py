# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("Fake Job Posting Detector — Harish Demo")
st.write("Paste a job description and the model will predict whether it's likely **REAL** or **FAKE/SCAM**.")

MODEL_PATH = "fake_job_detector.joblib"

# Load model
if not Path(MODEL_PATH).exists():
    st.error("Model not found. Run `python train_model.py` first to create the model.")
    st.stop()

model = joblib.load(MODEL_PATH)

def preprocess_text(text):
    t = text.strip()
    # simple cleaning
    t = re.sub(r"\s+", " ", t)
    return t

# Input box
job_text = st.text_area("Paste job description here", height=200)

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict"):
        if not job_text.strip():
            st.warning("Please paste a job description first.")
        else:
            text_proc = preprocess_text(job_text)
            proba = model.predict_proba([text_proc])[0]
            pred = model.predict([text_proc])[0]
            # proba[1] = prob of REAL (label 1)
            prob_real = proba[1]
            prob_fake = proba[0]

            if pred == 1:
                st.success(f"Prediction: **REAL** (confidence {prob_real*100:.1f}%)")
            else:
                st.error(f"Prediction: **FAKE / SCAM** (confidence {prob_fake*100:.1f}%)")

            # show probabilities
            st.write({"Real (1)": f"{prob_real*100:.1f}%", "Fake (0)": f"{prob_fake*100:.1f}%"})

            # show top contributing features (approx) using coef
            try:
                # works for linear models with vectorizer
                vec = model.named_steps["tfidf"]
                clf = model.named_steps["clf"]
                X_vec = vec.transform([text_proc])
                feature_names = np.array(vec.get_feature_names_out())
                coefs = clf.coef_[0]
                # multiply tfidf values with coefs to estimate contributions
                contrib = X_vec.toarray()[0] * coefs
                top_idx = np.argsort(contrib)[-8:][::-1]
                top_features = feature_names[top_idx]
                top_values = contrib[top_idx]
                st.markdown("**Top contributing tokens (approx):**")
                for tok, val in zip(top_features, top_values):
                    st.write(f"- {tok}  → {val:.3f}")
            except Exception as e:
                st.write("Feature contribution not available:", e)

with col2:
    if st.button("Example: Show sample FAKE"):
        st.write("**Fake sample:** Earn 50,000 per week working from home! No experience required, just pay a small training fee of $99 to start.")
    if st.button("Example: Show sample REAL"):
        st.write("**Real sample:** We are looking for a Software Engineer with 2+ years of experience in Python, REST APIs and AWS. Apply at careers@company.com")

st.markdown("---")
st.write("Tip: This is a simple prototype. To improve accuracy, train with a larger labelled dataset and tune classifier.")
