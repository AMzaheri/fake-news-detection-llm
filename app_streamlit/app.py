import streamlit as st
from tabs import predict_tab, disclaimer_tab, dashboard_tab

st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("Fake News Detection using LLM")

TABS = {
    "Predict": predict_tab.run,
    "Disclaimer": disclaimer_tab.run,
    "Dashboard (WIP)": dashboard_tab.run
}

tab = st.sidebar.radio("Navigation", list(TABS.keys()))
TABS[tab]()
