from transformers import pipeline

import os

# Define model path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, '..',\
                          'model', 'fine_tuned_model')

# Load the fine-tuned model
classifier = pipeline("text-classification",\
                       model=MODEL_PATH)

def predict_news(text):
    """
    Predict whether a news article is REAL or FAKE using the fine-tuned LLM.
    """
    result = classifier(text)[0]
    label = result['label']  # "LABEL_0" for FAKE or "LABEL_1" for REAL
    
    if label == "LABEL_0":
        return "FAKE"
    else:
        return "REAL"

