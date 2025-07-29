from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('../model/credit_risk_model.pkl')  # Make sure this matches the new model

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        feature_names = ['age', 'income', 'loan_amount', 'credit_score']
        inputs = [float(request.form.get(x)) for x in feature_names]
        
        # Wrap in a DataFrame with correct feature names
        input_df = pd.DataFrame([inputs], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
