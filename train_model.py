from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
data = load_iris()

# Create the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(data.data, data.target)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")