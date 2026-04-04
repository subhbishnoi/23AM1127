import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

mlflow.set_tracking_uri("mlruns")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n in [50, 100, 200]:
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"n={n} accuracy={acc:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model trained and saved!")