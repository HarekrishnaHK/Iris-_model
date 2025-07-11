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



# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route("/test", methods=["POST"])
# def test_input():
#     data = request.get_json()
#     print("✅ Received JSON input:")
#     print(data)

#     # You can also extract values if needed
#     sepal_length = data.get("sepal_length")
#     sepal_width = data.get("sepal_width")
#     petal_length = data.get("petal_length")
#     petal_width = data.get("petal_width")

#     return jsonify({
#         "message": "Data received successfully",
#         "received_data": {
#             "sepal_length": sepal_length,
#             "sepal_width": sepal_width,
#             "petal_length": petal_length,
#             "petal_width": petal_width
#         }
#     })

# if __name__ == "__main__":
#     app.run(debug=True, port=5001)
