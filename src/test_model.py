from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def test_model_accuracy():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    accuracy = model.score(X, y)
    assert accuracy > 0.9
    print(f"Accuracy: {accuracy}")