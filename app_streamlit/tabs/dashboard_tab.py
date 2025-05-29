import streamlit as st
import pandas as pd
import altair as alt


def run():
    st.header("Prediction Dashboard")

    if "history" not in st.session_state or not st.session_state.history:
        st.info("No predictions yet. Use the 'Predict' tab first.")
        return

    df = pd.DataFrame(st.session_state.history)

    st.subheader("Total Predictions")
    st.metric("Count", len(df))

    st.subheader("Prediction Distribution")
    pie_chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field="label", type="nominal", aggregate="count"),
        color=alt.Color(field="label", type="nominal"),
        tooltip=["label", "count()"]
    )
    st.altair_chart(pie_chart, use_container_width=True)

    # Calculate text lengths
    df["text_length"] = df["text"].apply(len)

    st.subheader("Length of News Texts")
    st.write("Average Length:", int(df["text_length"].mean()))

    hist = alt.Chart(df).mark_bar(color="salmon").encode(
           alt.X("text_length", bin=alt.Bin(maxbins=30), title="Text Length"),
           y='count()',
           tooltip=['count()']
        ).properties(
           width=600,
           height=300,
           title="Distribution of News Text Lengths"
        )
    st.altair_chart(hist, use_container_width=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    line_chart = alt.Chart(df).mark_line(point=True, color="orange").encode(
        x="timestamp:T",
        y="count()",
        tooltip=["timestamp:T", "count()"]
        ).properties(
            title="Prediction Activity Over Time"
    )

    st.subheader("Predictions Over Time")
    st.altair_chart(line_chart, use_container_width=True)

    st.success("Dashboard updated")
