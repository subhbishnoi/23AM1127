from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris ML API")
model = joblib.load("models/model.pkl")
iris_names = ["setosa", "versicolor", "virginica"]

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "ML API is running!"}

@app.post("/predict")
def predict(data: Features):
    X = np.array([[data.sepal_length, data.sepal_width,
                   data.petal_length, data.petal_width]])
    pred = model.predict(X)[0]
    return {"prediction": iris_names[pred], "class_id": int(pred)}