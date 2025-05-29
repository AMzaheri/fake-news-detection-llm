import os
import pandas as pd

os.environ["HOME"] = "/tmp"

import streamlit as st
from tabs import predict_tab, disclaimer_tab, dashboard_tab
import utils
#-------------------------------------------------
if "history" not in st.session_state:
    if os.path.exists(utils.LOG_PATH):
        st.session_state.history = pd.read_csv(utils.LOG_PATH).to_dict(orient="records")
    else:
        st.session_state.history = []

st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("Fake News Detection using LLM")

TABS = {
    "Predict": predict_tab.run,
    "Disclaimer": disclaimer_tab.run,
    "Dashboard (WIP)": dashboard_tab.run
}

tab = st.sidebar.radio("Navigation", list(TABS.keys()))
TABS[tab]()
