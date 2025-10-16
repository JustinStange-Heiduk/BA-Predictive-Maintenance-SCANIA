import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import confusion_matrix

# -------------------------
# Hilfsfunktionen für RSF
# -------------------------

def class_probs_from_S_tau(S_tau_row: np.ndarray) -> np.ndarray:
    """Berechnet die Klassenwahrscheinlichkeiten p₀–p₄ aus den Überlebensfunktionen S(t).

    Args:
        S_tau_row (np.ndarray): Überlebenswahrscheinlichkeiten an den Klassengrenzen,
            Vektor der Länge 4 [S1, S2, S3, S4].

    Returns:
        np.ndarray: Normalisierte Klassenwahrscheinlichkeiten der Länge 5
        (p0, p1, p2, p3, p4).
    """
    S1, S2, S3, S4 = S_tau_row.tolist()
    p4 = 1.0 - S1
    p3 = S1 - S2
    p2 = S2 - S3
    p1 = S3 - S4
    p0 = S4
    p = np.array([p0, p1, p2, p3, p4], dtype=float)
    p = np.clip(p, 0.0, 1.0)
    s = p.sum()
    return p / s if s > 0 else np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def decide_with_cost_from_rsf_at_taus(
    rsf, X: np.ndarray, taus: np.ndarray, cost: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entscheidet die RUL-Klasse basierend auf minimalen erwarteten Kosten.

    Args:
        rsf: Trainiertes Random Survival Forest Modell (scikit-survival).
        X (np.ndarray): Feature-Matrix (n_samples, n_features).
        taus (np.ndarray): Klassengrenzen (array der Länge 4).
        cost (np.ndarray): Kostenmatrix der Form (5, 5).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Vorhergesagte Klassen (n_samples,)
            - Minimale erwartete Kosten pro Sample (n_samples,)
            - Klassenwahrscheinlichkeiten p₀–p₄ (n_samples, 5)
    """
    surv_fns = rsf.predict_survival_function(X, return_array=False)
    N = len(X)
    taus = np.asarray(taus, dtype=float)
    S_tau = np.zeros((N, len(taus)), dtype=float)

    for i, fn in enumerate(surv_fns):
        vals = fn(taus)
        S_tau[i, :] = float(vals) if np.ndim(vals) == 0 else np.asarray(vals, dtype=float)

    probs = np.vstack([class_probs_from_S_tau(S_tau[i, :]) for i in range(N)])
    exp_cost_matrix = probs @ cost
    pred_class = exp_cost_matrix.argmin(axis=1).astype(int)
    exp_cost_min = exp_cost_matrix[np.arange(N), pred_class].astype(float)

    return pred_class, exp_cost_min, probs


def evaluate_decision_costs_from_true(
    true_class: np.ndarray, pred_class: np.ndarray, cost: np.ndarray
) -> Tuple[float, float, np.ndarray, float]:
    """Berechnet die realisierten Kosten und die Accuracy anhand der Vorhersagen.

    Args:
        true_class (np.ndarray): Wahre Klassenlabels (n_samples,).
        pred_class (np.ndarray): Vorhergesagte Klassen (n_samples,).
        cost (np.ndarray): Kostenmatrix (5, 5).

    Returns:
        Tuple[float, float, np.ndarray, float]:
            - Durchschnittliche Kosten pro Sample
            - Realisierte Gesamtkosten
            - Confusion Matrix (5x5)
            - Klassifikationsgenauigkeit (Accuracy)
    """
    n = np.asarray(true_class, dtype=int)
    m = np.asarray(pred_class, dtype=int)

    if n.shape != m.shape:
        raise ValueError("true_class und pred_class müssen gleich lang sein.")

    cm = confusion_matrix(n, m, labels=[0, 1, 2, 3, 4])
    realized_total = float(np.sum(cm * cost))
    avg_cost = realized_total / float(len(n))
    accuracy = float(np.mean(n == m))

    return avg_cost, realized_total, cm, accuracy


def decide_with_argmax_from_rsf_at_taus(
    rsf, X: np.ndarray, taus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entscheidet die RUL-Klasse basierend auf maximaler Wahrscheinlichkeit (Argmax).

    Args:
        rsf: Trainiertes Random Survival Forest Modell (scikit-survival).
        X (np.ndarray): Feature-Matrix (n_samples, n_features).
        taus (np.ndarray): Klassengrenzen (array der Länge 4).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Vorhergesagte Klassen (n_samples,)
            - Wahrscheinlichkeiten der gewählten Klassen (n_samples,)
            - Alle Klassenwahrscheinlichkeiten p₀–p₄ (n_samples, 5)
    """
    surv_fns = rsf.predict_survival_function(X, return_array=False)
    N = len(X)
    taus = np.asarray(taus, dtype=float)
    S_tau = np.zeros((N, len(taus)), dtype=float)

    for i, fn in enumerate(surv_fns):
        vals = fn(taus)
        S_tau[i, :] = float(vals) if np.ndim(vals) == 0 else np.asarray(vals, dtype=float)

    probs = np.vstack([class_probs_from_S_tau(S_tau[i, :]) for i in range(N)])
    pred_class = probs.argmax(axis=1).astype(int)
    max_probs = probs[np.arange(N), pred_class]

    return pred_class, max_probs, probs
