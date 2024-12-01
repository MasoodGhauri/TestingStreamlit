import os
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

# Load models and tokenizers
model_sa = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
tokenizer_sa = AutoTokenizer.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
model_hb = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Reviews-hasBug-BB")
tokenizer_hb = AutoTokenizer.from_pretrained("fyp-buglens/Reviews-hasBug-BB")

class ReviewInput(BaseModel):
    input_text: str

@app.post("/predict-hasbug")
async def predict_hasbug(review: ReviewInput):
    inputs = tokenizer_hb(review.input_text, return_tensors="pt")
    outputs = model_hb(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return {"result": predicted_class}

@app.post("/predict-sentiment")
async def predict_sentiment(review: ReviewInput):
    print(review.input_text)
    inputs = tokenizer_sa(review.input_text, return_tensors="pt")
    outputs = model_sa(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return {"result": predicted_class}

if __name__ == "__main__":
    # Get the port number from the environment variable (use 8000 if not set)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
