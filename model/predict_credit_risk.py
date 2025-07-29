import pandas as pd
import joblib

model = joblib.load('credit_risk_model.pkl')

# Example new input
new_data = pd.DataFrame([{
    'age': 45,
    'income': 65000,
    'credit_score': 720,
    'loan_amount': 12000,
    'loan_duration_months': 24,
    'employment_status': 0
}])

prediction = model.predict(new_data)
print("Prediction (0 = not defaulted, 1 = defaulted):", prediction[0])