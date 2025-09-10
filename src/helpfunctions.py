from typing import Dict, Any, Iterable, Tuple
import joblib
import os
import pickle
import xgboost as xgb
from pathlib import Path
import pandas as pd




# -------------------------
# Model loading
# -------------------------

def load_model_relative_to_script(rel_path: str):
    """Lädt ein Modell relativ zum Speicherort des aufrufenden Skripts (z.B. app.py)."""
    script_dir = Path(__file__).parent.resolve()
    model_path = (script_dir / rel_path).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")
    
    print(f"Lade Modell: {model_path.name}")
    return joblib.load(model_path)


# -------------------------
# Selected Feature loading
# -------------------------
def load_selected_features() -> pd.DataFrame:
    """Lädt die ausgewählten Features."""
    features_path = Path(__file__).parent / "../data/04_feature/selected_features_p_corr.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Ausgewählte Features nicht gefunden: {features_path}")
    return pd.read_parquet(features_path)
