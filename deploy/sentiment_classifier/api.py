from fastapi import Depends, FastAPI
from pydantic import BaseModel
import  uvicorn

from model import Model, get_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    score: float


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    sentiment, score = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, score=score
    )


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port = 8000)
