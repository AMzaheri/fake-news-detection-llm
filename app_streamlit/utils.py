import streamlit as st
from transformers import pipeline
from datetime import datetime

MODEL_ID = "afsanehm/fake-news-detection-llm"
LOG_PATH = "prediction_log.csv"
#------------------------------------------

@st.cache_resource  # Persist across Streamlit reruns
def load_pipeline():
    return pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)

#------------------------------------------
def predict(text):
    classifier = load_pipeline()
    result = classifier(text)[0]
    raw_label = result["label"]

    label_map = {
        "LABEL_0": "FAKE",
        "LABEL_1": "REAL"
    }

    #return label_map.get(raw_label, raw_label)
    final_label = label_map.get(raw_label, raw_label)
    record_prediction(final_label, text)
    return final_label

#------------------------------------------
#def record_prediction(label, text):
#    if "history" not in st.session_state:
#        st.session_state.history = []
#
#    st.session_state.history.append({
#        "timestamp": datetime.now().isoformat(),
#        "label": label,
#        "text": text,
#        "length": len(text.split())
#    })


def record_prediction(label, text):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "text": text,
        "length": len(text)
    }

    # keep in session_state for immediate dashboards
    st.session_state.setdefault("history", []).append(row)

    # append to CSV so it survives restarts
    header = not os.path.exists(LOG_PATH)
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=header, index=False)
#------------------------------------------
