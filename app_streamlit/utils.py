import streamlit as st
from transformers import pipeline

MODEL_ID = "afsanehm/fake-news-detection-llm"

@st.cache_resource  # Persist across Streamlit reruns
def load_pipeline():
    return pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)

def predict(text):
    classifier = load_pipeline()
    result = classifier(text)[0]
    return result['label']
