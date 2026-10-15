
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="ML Prediction API")

with open("model.pkl", "rb") as file:
    saved = pickle.load(file)

model = saved["model"]
target_names = saved["target_names"]


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: FlowerInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(features)[0])

    return {
        "prediction": str(target_names[prediction]),
        "class_id": prediction
    }
