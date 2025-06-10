import joblib
import sqlite3

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load model and scaler
model = joblib.load('iris_logistic_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Define class labels manually (since LabelEncoder was used)
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect('db.sqlite')
cursor = conn.cursor()

# Create table to store predictions
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

while True:
    try:
        print("\nEnter flower features (or type 'exit' to quit):")
        sl = input("Sepal Length: ")
        if sl.lower() == 'exit': break
        sw = input("Sepal Width: ")
        pl = input("Petal Length: ")
        pw = input("Petal Width: ")

        features = np.array([[float(sl), float(sw), float(pl), float(pw)]])
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        pred_label = class_labels[pred]

        print(f"🌸 Predicted Species: {pred_label}")

        # Save to database
        cursor.execute('''
            INSERT INTO predictions (sepal_length, sepal_width, petal_length, petal_width, predicted_species)
            VALUES (?, ?, ?, ?, ?)
        ''', (float(sl), float(sw), float(pl), float(pw), pred_label))
        conn.commit()

        print("✅ Saved to database.")
    except Exception as e:
        print("❌ Error:", e)

# Close DB connection when done
conn.close()
