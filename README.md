# 🚨 Fake Job Posting Detector (ML Project)

A Machine Learning web application that predicts whether a job posting is **REAL** or **FAKE/SCAM** using Natural Language Processing (NLP) and classification algorithms.

This project helps users identify fraudulent job ads that ask for money, personal details, or promise unrealistic salaries.

---

## 👨‍💻 Author
**Harish Yadav**  
Machine Learning | Python | Data Science

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- NLTK
- Streamlit

---

## 📂 Project Structure
Fake-job-detector/
│
├── app.py # Streamlit web app
├── train_model.py # Model training script
├── fake_job_detector.joblib # Trained model
├── requirements.txt # Project dependencies
└── README.md

---

## ⚙️ How It Works
1. Text is cleaned and vectorized using **TF-IDF**.
2. Logistic Regression classifies the job as **REAL (1)** or **FAKE (0)**.
3. Streamlit UI takes user input and shows:
   - Prediction
   - Confidence %
   - Top contributing words

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
Step 2: Train the model
python train_model.py
Step 3: Run the web app
streamlit run app.py

🧪 Example
Fake Job

Earn 50,000 per week working from home! No experience required. Pay small registration fee.

Real Job

We are hiring a Software Engineer with Python, AWS and REST API experience.

🚀 Future Improvements

Train on large real-world dataset

Use deep learning (LSTM / BERT)

Add login & database

Deploy on Streamlit Cloud
Open browser: http://localhost:8501
