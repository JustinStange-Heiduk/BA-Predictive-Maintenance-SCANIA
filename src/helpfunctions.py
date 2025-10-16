from typing import Dict, Any, Iterable, Tuple
import joblib
import os
import pickle
import xgboost as xgb
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np




# -------------------------
# Model loading
# -------------------------

def load_model_relative_to_script(rel_path: str):
    """
    Lädt ein mit joblib gespeichertes Modell relativ zum Speicherort des aufrufenden Skripts (z.B. app.py).

    Args:
        rel_path (str): Relativer Pfad zum Modell (z.B. "../data/06_models/RSF_final_model_test/model.joblib").

    Returns:
        Beliebiges Python-Objekt: Das geladene Modell.
    """
    script_dir = Path(__file__).parent.resolve()
    model_path = (script_dir / rel_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    print(f"Lade Modell: {model_path.name}")

    model = joblib.load(str(model_path))
    return model


# -------------------------
# Selected Feature loading
# -------------------------
def load_selected_features() -> pd.DataFrame:
    """Lädt die ausgewählten Features."""
    features_path = Path(__file__).parent / "../data/04_feature/selected_features_p_corr.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Ausgewählte Features nicht gefunden: {features_path}")
    return pd.read_parquet(features_path)


# -------------------------
# Bild  laden
# -------------------------
def load_scania_image():
    possible_paths = [
        Path(__file__).resolve().parent / "../data/08_reporting/scania.jpg",
        Path("/workspace/data/08_reporting/scania.jpg"),
    ]
    for path in possible_paths:
        if path.exists():
            return Image.open(path)
    return None




# -------------------------
# COST and TAUS
# -------------------------
def get_cost_and_taus()-> tuple[np.ndarray, np.ndarray]:
    """ Returns the cost matrix and class boundaries (taus) for RUL classification.    
    Kostenmatrix aus deinem Paper (Zeilen = Actual n, Spalten = Predicted m)

    Returns: tuple[np.ndarray, np.ndarray]: A tuple containing the cost matrix and class boundaries. 
    """
    COST = np.array([
        [0,   7,   8,   9,   10],
        [200, 0,   7,   8,    9],
        [300, 200, 0,   7,    8],
        [400, 300, 200, 0,    7],
        [500, 400, 300, 200,  0]
    ], dtype=float)

    # Klassengrenzen für RUL in Zeiteinheiten, konsistent zu deinen Labels 0..4
    # Beispiel: 4: [0,6), 3: [6,12), 2: [12,24), 1: [24,48), 0: [48, inf)
    TAUS = np.array([6.0, 12.0, 24.0, 48.0], dtype=float)

    return COST, TAUS