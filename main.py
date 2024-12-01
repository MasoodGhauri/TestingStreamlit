import streamlit as st
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from starlette.middleware.wsgi import WSGIMiddleware
from urllib.parse import urlparse

# FastAPI application
api_app = FastAPI()

# Load sentiment analysis model and tokenizer
model_sa = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
tokenizer_sa = AutoTokenizer.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")

# Load hasBug model and tokenizer
model_hb = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Reviews-hasBug-BB")
tokenizer_hb = AutoTokenizer.from_pretrained("fyp-buglens/Reviews-hasBug-BB")

# Pydantic model for input
class ReviewInput(BaseModel):
    input_text: str

# Route for predicting sentiment
@api_app.post("/predict-sentiment")
async def predict_sentiment(data: ReviewInput):
    inputs = tokenizer_sa(data.input_text, return_tensors="pt")
    outputs = model_sa(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return {"result": predicted_class}

# Route for predicting hasBug
@api_app.post("/predict-hasbug")
async def predict_hasbug(data: ReviewInput):
    inputs = tokenizer_hb(data.input_text, return_tensors="pt")
    outputs = model_hb(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return {"result": predicted_class}

# Integrate FastAPI with Streamlit
st.title("FastAPI + Streamlit Integration")

# Display FastAPI results on Streamlit
st.markdown("### Enter Text to Process")

# Input text for the API
input_text = st.text_input("Enter the text to analyze:")

# Endpoint selection
endpoint = st.selectbox("Select Endpoint", ["predict-sentiment", "predict-hasbug"])

if st.button("Submit"):
    if input_text:
        # Make a request to the FastAPI endpoint
        url = f"http://localhost:8000/{endpoint}"
        payload = {"input_text": input_text}

        try:
            # Send request to FastAPI
            response = st.experimental_get_query_params()["fetch"](url, payload)
            response.raise_for_status()

            # Display the result
            result = response.json()["result"]
            st.success(f"API Result: {result}")
            st.markdown(f'<div id="result">{result}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter text to analyze.")

# Run FastAPI app as a WSGI app
st.write("Starting FastAPI...")
st.experimental_rerun("./api/stapi)
