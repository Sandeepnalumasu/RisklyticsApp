from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# Load sample data
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model to 'model.pkl'
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Dummy model saved as model.pkl")
