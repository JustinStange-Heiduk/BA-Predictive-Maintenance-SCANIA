import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

from preprocessing import (
    interpolate_readout_df,
    compute_differences_per_vehicle_test,
    create_all_fixed_time_index_windows,
    extract_tsfresh_features,
    select_relevant_features
)

from decision_utils import (
    decide_with_cost_from_rsf_at_taus,
    decide_with_argmax_from_rsf_at_taus,
    class_probs_from_S_tau
)

from helpfunctions import (
    load_selected_features,
    load_model_relative_to_script,
    load_scania_image,
    get_cost_and_taus
)

from openai_utils import (
    explain_with_chatgpt,
    build_model_context
)

from local_feature_utils import local_feature_importance

# -------------------------
# Load Model & Config
# -------------------------
SELECTED_FEATURES = load_selected_features()
MODEL_PATH = "../data/06_models/RSF_final_model_test/rsf_model.joblib"
rsf_model = load_model_relative_to_script(MODEL_PATH)

COST, TAUS = get_cost_and_taus()

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(layout="wide", page_title="Predictive Maintenance Demo", page_icon="🚛")

st.title("🚛 Predictive Maintenance Demo App")
st.write("RSF-Modell zur Vorhersage der Restlebensdauer-Klasse auf Basis eines einzelnen Readouts.")

# --- Lokales Bild anzeigen ---
image = load_scania_image()
if image:
    st.image(image, caption="Scania LKW", width=400)
else:
    st.warning("📁 Bild konnte nicht geladen werden.")

# --- CSV Upload ---
uploaded_file = st.file_uploader("📤 Lade ein einzelnes Readout (CSV) hoch", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Validierung ---
    if len(df) != 1:
        st.error("❌ Die Datei darf nur **eine einzige Datenzeile** enthalten – für diesen PoC ist genau **ein Readout** erforderlich.")
        st.stop()

    st.subheader("📊 Rohdaten (1 Readout)")
    st.write(df.head())

    # --- Preprocessing ---
    df = interpolate_readout_df(df)
    st.subheader("📊 Interpolated Input")
    st.write(df.head())

    df = compute_differences_per_vehicle_test(df)
    st.subheader("📊 Differenced Input")
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
    st.write(f"Geschätzter Restlebensdauer: {pred}")

    # --- Erweiterte Entscheidungslogik ---
    pred_cost, cost_min, probs_cost = decide_with_cost_from_rsf_at_taus(rsf_model, X, taus=TAUS, cost=COST)
    pred_argmax, max_prob, probs_argmax = decide_with_argmax_from_rsf_at_taus(rsf_model, X, taus=TAUS)

    st.subheader("📈 Wahrscheinlichkeiten & Entscheidungen")
    st.markdown("**Wahrscheinlichkeiten p₀–p₄ pro Klasse:**")
    prob_df = pd.DataFrame(probs_cost, columns=[f"p{i}" for i in range(5)])
    st.dataframe(prob_df.style.format("{:.2%}"))

    st.markdown("**Entscheidung basierend auf minimalen erwarteten Kosten:**")
    st.write(f"→ Klasse: {pred_cost[0]}  |  Erwartete Kosten: {cost_min[0]:.3f}")

    st.markdown("**Entscheidung basierend auf höchster Wahrscheinlichkeit:**")
    st.write(f"→ Klasse: {pred_argmax[0]}  |  Wahrscheinlichkeit: {max_prob[0]:.2%}")

    # --- Balkendiagramm der Klassenwahrscheinlichkeiten ---
    st.markdown("**🎯 Visualisierung der Klassenwahrscheinlichkeiten**")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([f"p{i}" for i in range(5)], probs_cost[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wahrscheinlichkeit")
    ax.set_title("Wahrscheinlichkeiten pro RUL-Klasse")
    st.pyplot(fig)

    # --- Erklärung mit ChatGPT ---
    reporting_path = "workspace/data/08_reporting"
    context = build_model_context(
        X=X,
        pred_class=pred_cost[0],
        cost_min=cost_min[0],
        probs=probs_cost,
        method="cost",
        reporting_path=reporting_path,
        taus=TAUS,
        cost=COST
    )

    explanation = explain_with_chatgpt(context)

    st.subheader("💬 Chatbot-Erklärung")
    st.write(explanation)





