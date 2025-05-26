import streamlit as st

#def disclaimer_tab():
def run():
    st.subheader("Disclaimer")
    st.write("""
    - This app uses a transformer model fine-tuned on historical data from the **[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)**.
    - Predictions reflect **linguistic patterns**, not factual accuracy.
    - Do not use this as a truth verification tool.
    """)
