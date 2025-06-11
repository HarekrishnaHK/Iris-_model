from flask import Flask, request, jsonify
import joblib

# Load model, scaler, and label encoder
model = joblib.load("iris_logistic_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
label_encoder = joblib.load("iris_label_encoder.pkl")  

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]

        scaled = scaler.transform([features])
        prediction_index = model.predict(scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_index])[0]  # ✅ Decode

        return jsonify({
            "predicted_class_index": int(prediction_index),
            "predicted_class_label": prediction_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = 5001
    print(f"✅ Server running at http://127.0.0.1:{port}/predict")
    app.run(debug=True, port=port)
