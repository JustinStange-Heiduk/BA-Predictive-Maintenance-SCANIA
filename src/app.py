import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

from preprocessing import (
    interpolate_readout_df,
    compute_differences_per_vehicle_test,
    create_all_fixed_time_index_windows,
    extract_tsfresh_features,
    select_relevant_features
)

from helpfunctions import load_selected_features, load_model_relative_to_script

# # -------------------------
# # Load Model & Config
# # -------------------------

SELECTED_FEATURES = load_selected_features()

MODEL_PATH = "../data/06_models/RSF_final_model_test/rsf_model.joblib"
rsf_model = load_model_relative_to_script(MODEL_PATH)



# -------------------------
# Streamlit App
# -------------------------
st.title("Predictive Maintenance Demo App")
st.write("RSF Modell – Vorhersage der Restlebensdauer-Klasse mit XAI")

uploaded_file = st.file_uploader("Upload a readout CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Input")
    st.write(df.head())

    # --- Preprocessing ---
    df = interpolate_readout_df(df)
    st.subheader("📊 Interpolated Input")
    st.write(df.head())
    df = compute_differences_per_vehicle_test(df)
    st.subheader("Diff Data")
    st.write(df.head())
    windows = create_all_fixed_time_index_windows(df, window_sizes=[8])  
    st.subheader("🪟 Sliding Windows")
    st.write(windows.head())
    features = extract_tsfresh_features(windows, n_workers=2)
    X = select_relevant_features(features, SELECTED_FEATURES)
    
    st.subheader("✅ Processed Features")
    st.write(X.head())

    # --- Prediction ---
    pred = rsf_model.predict(X)
    st.subheader("🔮 Prediction")
    st.write(f"Vorhergesagte Klassen: {pred}")

    # --- SHAP Explanation ---
    explainer = shap.TreeExplainer(rsf_model)
    shap_values = explainer.shap_values(X)

    st.subheader("📈 SHAP Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)
