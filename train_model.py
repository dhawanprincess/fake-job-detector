# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
import os

# ensure nltk stopwords available
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download("stopwords", download_dir=nltk_data_dir)

# ----- SAMPLE dataset (expand later) -----
data = [
    # REAL job posts (1)
    ("We are looking for a Software Engineer with 2+ years of experience in Python, REST APIs and AWS. Apply at careers@company.com", 1),
    ("Join our team as a Data Analyst. Must know SQL and Excel. Competitive salary and benefits. Visit company website to apply.", 1),
    ("Marketing intern position at well-known FMCG. Internship stipend and certificate on completion. Send resume to hr@brand.com", 1),
    ("Full-time role: Frontend developer required with React, CSS, HTML skills. Office location: Gurgaon. Contact through official portal.", 1),
    ("Hiring: Senior ML Engineer. PhD/Masters preferred. Medical insurance and provident fund included. Apply via company portal.", 1),

    # FAKE / SCAM job posts (0)
    ("Earn 50,000 per week working from home! No experience required, just pay a small training fee of $99 to start.", 0),
    ("Congratulations! You have been shortlisted. Send your Aadhar and bank details to receive joining bonus immediately.", 0),
    ("Work from home data entry. Paid daily. First pay after registration fee. Email us on quickpay-scams@example.com", 0),
    ("We will give you a job if you buy starter kit (~2000). Very easy work. Only few positions left. Contact immediately.", 0),
    ("Urgent hiring: Send your PAN and UPI details to receive salary advance. Immediate joining after small verification fee.", 0),
]

df = pd.DataFrame(data, columns=["text", "label"])
print("Sample dataset:")
print(df.head(10))

# ----- train/test split -----
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# ----- pipeline -----
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)),
    ("clf", LogisticRegression(solver="liblinear"))
])

# ----- train -----
pipeline.fit(X_train, y_train)

# ----- eval -----
y_pred = pipeline.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ----- save model -----
model_path = "fake_job_detector.joblib"
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")
