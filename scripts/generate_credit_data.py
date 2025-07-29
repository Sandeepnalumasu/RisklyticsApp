import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 1000
age = np.random.randint(18, 70, N)
income = np.random.randint(20000, 120000, N)
loan_amount = np.random.randint(1000, 50000, N)
credit_score = np.random.randint(300, 850, N)

# Create a risk score combining these features (just an example)
risk_score = (
    (loan_amount / income) * 0.6 +
    (700 - credit_score) * 0.3 +
    (70 - age) * 0.1
)

# Normalize risk score between 0 and 1
risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

# Assign default with higher chance if risk_score > 0.5
defaulted = np.where(risk_score > 0.5, np.random.choice([0, 1], N, p=[0.3, 0.7]), 
                              np.random.choice([0, 1], N, p=[0.9, 0.1]))

df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'defaulted': defaulted
})

output_dir = '../data/synthetic_credit_data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'training_data.csv')
df.to_csv(output_path, index=False)

print(f"Training data saved to {output_path}")
