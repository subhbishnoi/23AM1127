from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib, os

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model trained and saved!")