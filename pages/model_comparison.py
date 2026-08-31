import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Model Comparison")

results_path = "models/model_comparison.csv"

if not os.path.exists(results_path):
    st.error("Model comparison results were not found.")
    st.info("Please run train_model.py first.")
    st.stop()

results_df = pd.read_csv(results_path)

st.subheader("📊 Model Performance Comparison")

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)