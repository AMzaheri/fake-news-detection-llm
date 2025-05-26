#import streamlit as st
#from app_streamlit.utils import predict

#def predict_tab():
#    st.subheader("Check if a news article is REAL or FAKE")
#    text = st.text_area("Paste the news text:")
#    if st.button("Predict") and text:
#        label = predict(text)
#        st.markdown(f"**Prediction:** `{label}`")


import streamlit as st
from app_streamlit.utils import predict  # or adjust the import

def run():
    st.header("Check if a news article is REAL or FAKE")
    text = st.text_area("Paste news article")
    if st.button("Predict"):
        label = predict(text)
        st.success(f"Prediction: {label}")

