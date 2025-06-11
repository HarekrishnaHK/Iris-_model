from flask import Flask, request, jsonify
import pickle

# Load your saved Logistic Regression model and scaler
with open("iris_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("iris_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract input features from the request JSON
        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]

        # Scale the input features
        scaled = scaler.transform([features])

        # Predict the class
        prediction = model.predict(scaled)

        # You can also decode the class label if needed
        return jsonify({"predicted_class_index": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = 5001
    print(f"✅ Server running at http://127.0.0.1:{port}/predict")
    app.run(debug=True, port=port)
