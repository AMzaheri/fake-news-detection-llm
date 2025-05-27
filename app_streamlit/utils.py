import streamlit as st
from transformers import pipeline

MODEL_ID = "afsanehm/fake-news-detection-llm"

@st.cache_resource  # Persist across Streamlit reruns
def load_pipeline():
    return pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)

#def predict(text):
#    classifier = load_pipeline()
#    result = classifier(text)[0]
#    return result['label']

def predict(text):
    classifier = load_pipeline()
    result = classifier(text)[0]
    raw_label = result["label"]

    label_map = {
        "LABEL_0": "FAKE",
        "LABEL_1": "REAL"
    }

    return label_map.get(raw_label, raw_label)  # fallback to raw label if unknown

