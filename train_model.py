import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# =========================
# 1. Load dataset
# =========================

DATA_PATH = "data/fake_job_postings.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# =========================
# 2. Prepare text
# =========================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Replace missing values with empty strings
for col in text_columns:
    df[col] = df[col].fillna("")

# Combine all useful text fields
df["text"] = df[text_columns].agg(" ".join, axis=1)

# Target:
# 0 = REAL
# 1 = FAKE / FRAUDULENT
y = df["fraudulent"]
X = df["text"]


# =========================
# 3. Show class distribution
# =========================

print("\nClass distribution:")
print(y.value_counts())

print("\n0 = REAL")
print("1 = FAKE / FRAUDULENT")


# =========================
# 4. Train/Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# =========================
# 5. ML Pipeline
# =========================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )
    )
])


# =========================
# 6. Train
# =========================

print("\nTraining model...")
print("This may take a few minutes on an i3 laptop.")

pipeline.fit(X_train, y_train)


# =========================
# 7. Evaluation
# =========================

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["REAL", "FAKE"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 8. Save model
# =========================

MODEL_PATH = "fake_job_detector.joblib"

joblib.dump(pipeline, MODEL_PATH)

print("\n==============================")
print(f"Model saved to: {MODEL_PATH}")
print("==============================")