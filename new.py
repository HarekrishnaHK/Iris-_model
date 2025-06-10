import joblib
import sqlite3
import numpy as np

# Load saved model, scaler, and label encoder
model = joblib.load('iris_logistic_model.pkl')
scaler = joblib.load('iris_scaler.pkl')
label_encoder = joblib.load('iris_label_encoder.pkl')

# Connect to SQLite database
conn = sqlite3.connect('db.sqlite')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sepal_length REAL,
        sepal_width REAL,
        petal_length REAL,
        petal_width REAL,
        predicted_species TEXT
    )
''')
conn.commit()

# Continuous prediction loop
while True:
    try:
        print("\nEnter flower features (or type 'exit' to quit):")
        sl = input("Sepal Length: ")
        if sl.lower() == 'exit': break
        sw = input("Sepal Width: ")
        pl = input("Petal Length: ")
        pw = input("Petal Width: ")

        # Convert to float and scale
        features = np.array([[float(sl), float(sw), float(pl), float(pw)]])
        features_scaled = scaler.transform(features)

        # Predict
        pred = model.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        print(f"🌸 Predicted Species: {pred_label}")

        # Save to DB
        cursor.execute('''
            INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, predicted_species)
            VALUES (?, ?, ?, ?, ?)
        ''', (float(sl), float(sw), float(pl), float(pw), pred_label))
        conn.commit()
        print("✅ Saved to database.")
    except Exception as e:
        print("❌ Error:", e)

# Close connection
conn.close()
