import io
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

MODEL_PATH = "savedmodel.pth"

app = Flask(__name__)

print("📦 Loading model for Flask app...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

def preprocess_image(file_storage):
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes))

    image = image.convert("L")
    image = image.resize((64, 64))

    img_array = np.array(image, dtype="float32")
    img_flat = img_array.flatten() / 255.0
    img_flat = img_flat.reshape(1, -1)
    return img_flat

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    try:
        input_data = preprocess_image(file)
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            confidence = float(np.max(proba)) * 100.0
        else:
            confidence = None

        response = {
            "success": True,
            "predicted_person": int(prediction)
        }
        if confidence is not None:
            response["confidence"] = f"{confidence:.2f}%"

        return jsonify(response)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
