import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from preprocess import preprocess
from config import MODEL_PATH, VECTORIZER_PATH

df = preprocess(load_data())

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(MODEL_PATH)
vec = joblib.load(VECTORIZER_PATH)

X_test_vec = vec.transform(X_test)

preds = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))