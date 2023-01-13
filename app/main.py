# dependencies
from fastapi import FastAPI
from pydantic import BaseModel
from model.utils import predict_ld, predict_oe

# setup app
app = FastAPI()

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    classification: str
    probability: float

# define app methods
@app.get('/')
def home():
    out = {'health check': 'OK'}
    return out

@app.post('/predict_language', response_model=PredictionOutput)
def predict(input: TextInput):
    res = predict_ld(input.text)
    out = {'classification': res[0], 'probability': res[1]}
    return out

@app.post('/predict_offensive', response_model=PredictionOutput)
def predict(input: TextInput):
    res = predict_oe(input.text)
    out = {'classification': res[0], 'probability': res[1]}
    return out