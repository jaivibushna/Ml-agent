
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load sample dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Save trained model and class names
with open("model.pkl", "wb") as file:
    pickle.dump(
        {"model": model, "target_names": iris.target_names},
        file
    )

print("Model trained and saved successfully!")
