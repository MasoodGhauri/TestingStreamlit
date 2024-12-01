import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from urllib.parse import urlparse, parse_qs

# Load sentiment analysis model and tokenizer
model_sa = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")
tokenizer_sa = AutoTokenizer.from_pretrained("fyp-buglens/Review-SentimentAnalysis-BB")

# Load hasBug model and tokenizer
model_hb = AutoModelForSequenceClassification.from_pretrained("fyp-buglens/Reviews-hasBug-BB")
tokenizer_hb = AutoTokenizer.from_pretrained("fyp-buglens/Reviews-hasBug-BB")

# Helper functions for predictions
def predict_sentiment(text):
    """Predict sentiment of the input text using the sentiment analysis model."""
    inputs = tokenizer_sa(text, return_tensors="pt")
    outputs = model_sa(**inputs)
    return outputs.logits.argmax().item()

def predict_hasbug(text):
    """Predict if the input text indicates a bug using the hasBug model."""
    inputs = tokenizer_hb(text, return_tensors="pt")
    outputs = model_hb(**inputs)
    return outputs.logits.argmax().item()

def get_query_params():
    """Get query parameters from the URL."""
    url = st.experimental_get_url()
    query_string = urlparse(url).query
    return parse_qs(query_string)

# Streamlit app
st.title("Simulated RESTful API with Streamlit")

# Parse query parameters
query_params = get_query_params()

# Check if `input_text` is provided in the query parameters
if "input_text" in query_params:
    input_text = query_params["input_text"][0]
    
    # Check the endpoint specified in the query parameters
    if "endpoint" in query_params:
        endpoint = query_params["endpoint"][0]

        if endpoint == "predict-sentiment":
            # Sentiment Analysis Prediction
            result = predict_sentiment(input_text)
            st.json({"result": result})
        elif endpoint == "predict-hasbug":
            # Bug Detection Prediction
            result = predict_hasbug(input_text)
            st.json({"result": result})
        else:
            st.json({"error": "Invalid endpoint specified"})
    else:
        st.json({"error": "No endpoint specified in the query parameters"})
else:
    st.write("Please provide `input_text` and `endpoint` as query parameters in the URL.")

