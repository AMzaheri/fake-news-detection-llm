from transformers import pipeline

import os

# Define model path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fine_tuned_model")
#PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(PROJECT_ROOT, '..',\
#                          'model', 'fine_tuned_model')


print("MODEL_PATH =", MODEL_PATH)
print("Contents of fine_tuned_model:")
print(os.listdir(MODEL_PATH))

# load from local directory only
#classifier = pipeline(
#    "text-classification",
#    model=MODEL_PATH,
#    tokenizer=MODEL_PATH,
#    local_files_only=True  # NOT go to the hub
#    )
# load the model from Hugging Face afsanehm/fake-news-detection-llm
classifier = pipeline(
    "text-classification",
    model="afsanehm/fake-news-detection-llm",  # your HF repo
    tokenizer="afsanehm/fake-news-detection-llm"
)

#----------------------------------
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

