import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset (adjust the path as needed)
df = pd.read_csv('../data/synthetic_credit_data/training_data.csv')

# Use only the 4 features you mentioned
features = ['age', 'income', 'loan_amount', 'credit_score']
X = df[features]
y = df['defaulted']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'credit_risk_model.pkl')

print("✅ Model trained and saved as 'credit_risk_model.pkl'")
