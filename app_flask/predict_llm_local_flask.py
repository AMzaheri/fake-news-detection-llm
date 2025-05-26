# app/predict_llm_local_flask.py (for local Flask app)

from transformers import pipeline
import os

# Define model path relative to this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fine_tuned_model")

# Load classifier locally
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    local_files_only=True
)

def predict_news(text):
    result = classifier(text)[0]
    return "REAL" if result['label'] == 'LABEL_1' else "FAKE"

