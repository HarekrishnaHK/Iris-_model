import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('iris.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Print column names and shape of features
print("🔍 Columns:", df.columns.tolist())

# Select only feature columns
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].values

# Check shapes
print("✅ Shape of X (before scaling):", X.shape)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Shape of X_scaled:", X_scaled.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build and train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model, scaler, and label encoder
joblib.dump(model, 'iris_logistic_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
joblib.dump(label_encoder, 'iris_label_encoder.pkl')
print("✅ Model, scaler, and label encoder saved.")
