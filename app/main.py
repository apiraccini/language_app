# dependencies
from fastapi import FastAPI
from pydantic import BaseModel
from model.utils import predict_model

# setup app
app = FastAPI()

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    language: str

# define app methods
@app.get('/')
def home():
    out = {'health check': 'OK'}
    return out

@app.post('/predict', response_model=PredictionOutput)
def predict(input: TextInput):
    language = predict_model(input.text)
    out = {'language': language}
    return out