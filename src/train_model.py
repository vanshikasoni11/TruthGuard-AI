import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

fake = pd.read_csv(DATA_DIR / "Fake.csv")
true = pd.read_csv(DATA_DIR / "True.csv")

fake["label"] = 1      
true["label"] = 0      

df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()
df['text'] = df['text'].apply(clean_text)

df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42,stratify=df['label']
)
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, OUTPUT_DIR / "model.joblib")
joblib.dump(vectorizer, OUTPUT_DIR / "vectorizer.joblib")

pipeline = {
    "model": model,
    "vectorizer": vectorizer
}

joblib.dump(pipeline, OUTPUT_DIR / "pipeline.joblib")

print("✅ Model trained and saved")