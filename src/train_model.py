import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load cleaned dataset
data = pd.read_csv("data/cleaned_upi_transactions.csv")

# Target column
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/fraud_model.pkl")

print("Model trained and saved successfully.")
