from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from pathlib import Path

# Client initialisieren (Key aus ENV lesen)

load_dotenv(dotenv_path="env/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------------------
# Chatbot-Context
# -------------------------------------------------

def build_model_context(X: pd.DataFrame, pred_class: int, cost_min: float,
                        probs: np.ndarray, method: str,
                        reporting_path: str, taus: np.ndarray, cost: np.ndarray) -> str:
    """
    Baut den erklärenden Kontext für GPT auf.
    """
    # Features des aktuellen Readouts
    input_features = X.iloc[0].to_dict()

    # Wahrscheinlichkeit pro Klasse
    probs_dict = {f"p{i}": float(probs[0][i]) for i in range(len(probs[0]))}


    feature_importance_path = Path(__file__).parent / "../data/08_reporting/feature_test_perm_importance.parquet"
    if not feature_importance_path.exists():
        raise FileNotFoundError(f"Permutation Importance nicht gefunden: {feature_importance_path}")

    feat_imp_df = pd.read_parquet(feature_importance_path)
    feat_imp_df.index.name = "feature"
    top_features = feat_imp_df.head(10).to_dict()["importances_mean"]

    # Kontext bauen
    context = f"""
    Du bist ein technischer Assistent für Predictive Maintenance.
    Erkläre bitte eine Modellentscheidung für die Bachelorarbeit, die auf
    SCANIA Truck Sensordaten basiert.

    Projekt-Setup:
    - Datenquelle: SCANIA Component X Dataset (zeitabhängige Sensor-Readouts von Fahrzeugen)
    - Preprocessing: Interpolation + Differenzenbildung
    - Feature Engineering: Fixed Time Index Sliding Windows und tsfresh Feature-Extraktion
    - Modell: Random Survival Forest (RSF)
    - Klassifikation: Restlebensdauer (Remaining Useful Life, RUL) in 5 Klassen
    - Entscheidungsgrundlage: Kostenmatrix (falsch-positive vs. falsch-negative Fehler) und Wahrscheinlichkeiten

    Aktueller Vorhersagefall:
    - Vorhersageklasse (RUL): {pred_class}
    - Entscheidungslogik: {method}
    - Erwartete Kosten: {cost_min:.3f}
    - Wahrscheinlichkeiten p0–p4: {probs_dict}
    - Klassengrenzen (TAUs): {taus.tolist()}
    - Kostenmatrix (5x5): {cost.tolist()}

    Eingabefeatures des aktuellen Readouts:
    {input_features}

    Wichtigste Features laut Permutation Importance:
    {top_features}

    Aufgabe:
    Erkläre auf nachvollziehbare, technische Weise,
    warum das Modell diese Entscheidung getroffen hat.
    Gehe auf die Rolle der wichtigsten Features, die Bedeutung der Wahrscheinlichkeiten
    und den Einfluss der Kostenmatrix ein.

    Anleitung:
    1. Beschreibe die Rolle der wichtigsten Features. Gehe dabei auf die Art des tsfresh-Features ein (z.B. maximum, median) und erkläre, warum diese für Verschleiß oder RUL relevant sind.
    2. Erkläre, wie die Wahrscheinlichkeiten p0–p4 aus der Überlebensfunktion interpretiert werden.
    3. Zeige, wie die Kostenmatrix die Entscheidung beeinflusst (Minimierung erwarteter Kosten).
    4. Verknüpfe diese Aspekte zu einer konsistenten Erklärung für die Entscheidung auf Klasse {pred_class}.
    5. Achte auf wissenschaftlich-technischen Stil, geeignet für eine Bachelorarbeit.
    """
    return context


# -------------------------------------------------
# Chatbot-Erklärer
# -------------------------------------------------
def explain_with_chatgpt(context: str) -> str:
    """
    Fragt ChatGPT nach einer Erklärung auf Basis des Kontextes.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du bist ein technischer Assistent, der Modellentscheidungen in Predictive Maintenance erklärt."},
            {"role": "user", "content": context}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
