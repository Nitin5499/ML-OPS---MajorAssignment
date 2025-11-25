import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "savedmodel.pth"

def load_data():
    print("📥 Loading Olivetti faces dataset for testing...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def load_model():
    print(f"📦 Loading model from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)
    return model

def main():
    _, X_test, _, y_test = load_data()
    model = load_model()

    print("🔍 Evaluating model on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
