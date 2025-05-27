import streamlit as st

#def disclaimer_tab():
def run():
    st.subheader("Disclaimer")
    st.write("""
- This app uses a transformer model fine-tuned on historical data from the **[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)**.
- The training process and results are documented in my [Kaggle notebook](https://www.kaggle.com/code/afsanehm/fake-news-detection-with-llm-fine-tuning).
- The fine-tuned model is available on Hugging Face: [Model on HF](https://huggingface.co/afsanehm/fake-news-detection-llm)

---

### Important Observation

During testing, I found the model tends to classify:
- **Short texts** (e.g. headlines) as **FAKE**
- **Longer, structured articles** as **REAL**

This is likely because:
- The ISOT dataset contains **short fake news** and **long real news**
- So the model has learned **text length and structure** as proxies for truthfulness. This is not true semantic understanding.

Please interpret results **with caution**.
- Predictions reflect **linguistic patterns**, not factual accuracy.
- Do not use this app as a truth verification tool.
    """)
