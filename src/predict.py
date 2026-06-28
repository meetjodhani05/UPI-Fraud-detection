import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

def predict_transaction(transaction):
    data = pd.DataFrame([transaction])
    prediction = model.predict(data)

    if prediction[0] == 1:
        return "Fraud Transaction"
    else:
        return "Safe Transaction"
