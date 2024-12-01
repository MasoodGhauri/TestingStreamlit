import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# Load sentiment analysis model and tokenizer
model_sa = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
tokenizer_sa = AutoTokenizer.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")

# Load hasBug model and tokenizer
model_hb = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Reviews-hasBug-BB")
tokenizer_hb = AutoTokenizer.from_pretrained("fyp-buglens/Reviews-hasBug-BB")

# Helper function for predictions
def predict_sentiment(text):
    inputs = tokenizer_sa(text, return_tensors="pt")
    outputs = model_sa(**inputs)
    return outputs.logits.argmax().item()

def predict_hasbug(text):
    inputs = tokenizer_hb(text, return_tensors="pt")
    outputs = model_hb(**inputs)
    return outputs.logits.argmax().item()

# Streamlit app
st.title("RESTful API Simulation with Streamlit")

# Expose as "RESTful-like API"
# Parse query params for API simulation
query_params = st.experimental_get_query_params()

if "input_text" in query_params:
    input_text = query_params["input_text"][0]
    
    # Check the requested endpoint
    if "endpoint" in query_params:
        endpoint = query_params["endpoint"][0]
        if endpoint == "predict-sentiment":
            result = predict_sentiment(input_text)
            st.json({"result": result})
        elif endpoint == "predict-hasbug":
            result = predict_hasbug(input_text)
            st.json({"result": result})
        else:
            st.json({"error": "Invalid endpoint"})
    else:
        st.json({"error": "No endpoint specified"})
else:
    st.write("Use query parameters to send API requests.")
