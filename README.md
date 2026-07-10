# 🛡️ TruthGuard AI

An AI-powered Hybrid Fake News Detection System that combines fine-tuned **DistilBERT**, **Google Fact Check API**, and **NewsAPI** to classify and verify news articles in real time. Built with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**.

---

# 📑 Table of Contents

- Overview
- Features
- Demo
- Project Structure
- Installation
- Dataset
- Model Training
- Detection Pipeline
- Running the Application
- Technologies Used
- Future Improvements
- Author
- License

---

# 📖 Overview

TruthGuard AI is an end-to-end NLP application designed to detect fake news using a hybrid approach.

Instead of relying only on machine learning predictions, the system combines:

- Fine-tuned DistilBERT
- Google Fact Check API
- NewsAPI
- Confidence-based decision logic

to provide evidence-backed predictions with real-time verification.

---

# 🚀 Features

- Fine-tuned DistilBERT for fake news classification
- Hybrid verification using Google Fact Check API
- Latest news validation using NewsAPI
- Confidence scores for predictions
- Explainable prediction messages
- Interactive Streamlit web application
- Secure API key management using `.env`
- Research publication based project

---

# 🎯 Demo

## Live Application

https://fake-news-detector-uwxww795wdtr9hlwbgusms.streamlit.app/

### Example Features

- Detect REAL or FAKE news
- Confidence visualization
- Google Fact Check verification
- Latest related news
- Explanation of prediction

screenshots

---

# 📂 Project Structure

```text
TruthGuard-AI/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│
├── outputs/
│   ├── bert_model/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   ├── metrics.json
│
├── src/
│   ├── train_bert.py
│   ├── predict_bert.py
│   ├── streamlit_app.py
│   ├── hybrid.py
│   ├── fact_check.py
│   ├── news_api.py
│   ├── preprocess.py
│   ├── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/vanshikasoni11/TruthGuard-AI

cd TruthGuard-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📊 Dataset

Dataset Used:

Fake and Real News Dataset (Kaggle)

Total Articles:

- 44,898+ news articles
- Binary Classification
- REAL
- FAKE

The dataset was cleaned, preprocessed, tokenized, and used for fine-tuning DistilBERT.

---

# 🤖 Model Training

Train DistilBERT

```bash
python src/train_bert.py
```

The training script:

- Loads the dataset
- Tokenizes text using DistilBERT tokenizer
- Fine-tunes DistilBERT
- Evaluates the model
- Saves trained weights

Saved Model

```
outputs/bert_model/
```

---

# 📈 Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **83%** |
| Model | DistilBERT |
| Framework | Hugging Face Transformers |
| Dataset | 44,898+ Articles |

> Performance is measured on a held-out test dataset. Real-world news may vary in complexity and uncertainty.

---

# 🔍 Detection Pipeline

```text
User Input
      │
      ▼
Text Cleaning
      │
      ▼
DistilBERT Prediction
      │
      ▼
Google Fact Check API
      │
      ▼
NewsAPI Verification
      │
      ▼
Hybrid Decision Engine
      │
      ▼
REAL / FAKE / UNCERTAIN
```

---

# ▶️ Running the Application

Launch Streamlit

```bash
streamlit run src/streamlit_app.py
```

Open

```
http://localhost:8501
```

---

# 💡 Application Features

- News classification
- Confidence score
- Explainable prediction
- Google Fact Check integration
- NewsAPI integration
- Real-time inference
- Responsive Streamlit interface

---

# 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Streamlit
- Google Fact Check API
- NewsAPI
- Pandas
- NumPy
- Scikit-learn
- Git
- GitHub

---

# 🔮 Future Improvements

- Multi-language fake news detection
- Explainable AI (SHAP/LIME)
- Image and video misinformation detection
- Multimodal transformer models
- User authentication and history
- Docker deployment

---

# 👩‍💻 Author

**Vanshika Soni**

B.Tech Computer Science Engineering

Published Researcher | AI & Full Stack Developer

GitHub: https://github.com/vanshikasoni11

---

# 📄 Research Publication

**TruthGuard AI v2: A Hybrid Fake News Detection System Using ML and Transformer-Based NLP**

Published in **IJRPR (ISSN 2582-7421)**

---

# 📜 License

This project is licensed under the MIT License.
