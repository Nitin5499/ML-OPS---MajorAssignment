import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data():
    print("📥 Loading Olivetti faces dataset...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target

    print(f"Total samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("🧠 Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier(
        max_depth=20,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"✅ Training Accuracy: {train_acc * 100:.2f}%")

    model_path = "savedmodel.pth"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")

if __name__ == "__main__":
    main()
