# Neue Datei: local_feature_utils.py
import pandas as pd
import numpy as np

from decision_utils import class_probs_from_S_tau



# -------------------------
# feature importance
# -------------------------

def local_feature_importance(
    rsf_model,
    X: pd.DataFrame,
    instance: pd.Series,
    taus: np.ndarray
) -> pd.DataFrame:
    """
    Lokale Feature-Wichtigkeit: Für jedes Feature schauen wir,
    wie stark sich die Klassenwahrscheinlichkeiten ändern,
    wenn der Wert auf den Median gesetzt wird.
    """
    instance = instance.copy()

    # Survival-Funktion für Original-Readout
    surv_fn = rsf_model.predict_survival_function(instance.to_frame().T, return_array=False)[0]
    base_S = np.array([surv_fn(tau) for tau in taus])  # Länge == len(taus) == 4
    base_probs = class_probs_from_S_tau(base_S)

    impacts = []
    for col in X.columns:
        modified = instance.copy()
        modified[col] = X[col].median()

        mod_fn = rsf_model.predict_survival_function(modified.to_frame().T, return_array=False)[0]
        mod_S = np.array([mod_fn(tau) for tau in taus])
        mod_probs = class_probs_from_S_tau(mod_S)

        # Unterschied der Wahrscheinlichkeiten messen
        diff = np.abs(base_probs - mod_probs).sum()
        impacts.append((col, diff))

    df = pd.DataFrame(impacts, columns=["Feature", "Impact"])
    return df.sort_values("Impact", ascending=False).reset_index(drop=True)


