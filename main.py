from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

app = FastAPI()

# Load hasBug model and tokenizer
model_sa = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
tokenizer_sa = AutoTokenizer.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")

# Load hasBug model and tokenizer
model_hb = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Reviews-hasBug-BB")
tokenizer_hb = AutoTokenizer.from_pretrained("fyp-buglens/Reviews-hasBug-BB")

# Define a Pydantic model for the request body
class ReviewInput(BaseModel):
    input_text: str

# route for hasbug model prediction
@app.post("/predict-hasbug")
async def predict(review: ReviewInput):
    # Tokenize the input text
    inputs = tokenizer_hb(review.input_text, return_tensors="pt")
    
    # Pass the tokenized inputs to the model
    outputs = model_hb(**inputs)
    
    # Get the predicted class (logits)
    predicted_class = outputs.logits.argmax().item()
    
    # Return the prediction result
    st.write(predicted_class)


# route for sentiment analysis model prediction
@app.post("/predict-sentiment")
async def predict(review: ReviewInput):
    # Tokenize the input text
    inputs = tokenizer_sa(review.input_text, return_tensors="pt")
    
    # Pass the tokenized inputs to the model
    outputs = model_sa(**inputs)
    
    # Get the predicted class (logits)
    predicted_class = outputs.logits.argmax().item()
    
    # Return the prediction result
    st.write(predicted_class)
