# SCANIA Component X – Predictive Maintenance Projekt

## 1. Überblick

Dieses Projekt implementiert einen vollständigen **Predictive Maintenance Workflow** für das **SCANIA Component X Dataset**.  
Ziel ist die **Vorhersage der Restlebensdauer (Remaining Useful Life, RUL)** von Fahrzeugkomponenten mittels **Survival-Analyse** und **klassischer Machine-Learning-Modelle**.

Besonderheiten:
- Zeitreihenbasiertes Feature Engineering mit **Sliding Windows** und **tsfresh**
- Survival-Modelle: **Random Survival Forest (RSF)**, **XGBoost AFT**
- Modellbewertung mit **Kostenmatrix** (realistische Kosten für Fehleinschätzungen)
- Experiment-Tracking via **MLflow**
- Interaktive **Streamlit-App** für Deployment
- Explainability mit **Permutation Importance** und **GPT-basierter natürlicher Spracherklärung**

---

## 2. Datenquelle

Die Daten stammen aus dem öffentlich zugänglichen Forschungskatalog des  
**Swedish National Data Service (SND)**:

> **Titel:** SCANIA Component X Dataset  
> **Quelle:** [researchdata.se – Dataset 2024-34/2](https://researchdata.se/en/catalogue/dataset/2024-34/2)  
> **Zugriffsdatum:** 14. Juli 2025  
> **Lizenz:** CC BY 4.0 (Creative Commons Attribution)

Das Dataset umfasst neun CSV-Dateien für Training, Validierung und Test:

```text
data/01_raw/
├── train_specifications.csv
├── train_operational_readouts.csv
├── train_tte.csv
├── validation_specifications.csv
├── validation_operational_readouts.csv
├── validation_labels.csv
├── test_specifications.csv
├── test_operational_readouts.csv
├── test_labels.csv
```
---

## 3. Daten-Pipeline

Die Daten werden in mehreren Stufen verarbeitet:

| Ordner             | Zweck                                                                 | Beispiele                           |
|--------------------|----------------------------------------------------------------------|------------------------------------|
| **01_raw**         | Unveränderte Rohdaten                                                | Original CSV-Dateien von SND        |
| **02_intermediate**| Vorverarbeitete Daten (Interpolation, Differenzenbildung, `dropna`)  | interpolated_readouts.parquet       |
| **03_primary**     | Integrierte, bereinigte Struktur (Window-basiert)                    | windows_8.csv, windows_32.csv       |
| **04_feature**     | Von **tsfresh** generierte Features                                  | features_window8.csv, features_window32.csv |
| **05_model_input** | Finalisierte Inputs für Modelle nach Feature Selection               | X_train.parquet, X_val.parquet      |
| **06_models**      | Gespeicherte Modelle                                                 | rsf_model.joblib, aft_model.json    |
| **07_model_output**| Modelloutputs (Predictions, Residuen, Wahrscheinlichkeiten)          | y_pred.csv, rul_pred.csv, shap.csv  |
| **08_reporting**   | Reports, Plots, Explainability                                       | ROC-Curve.png, feature_importance.csv |

---

## 4. Modellierung

### Feature Engineering
- Sliding Windows (Größe: 8)
- **tsfresh** Feature-Extraktion (MinimalFCParameters)
- Feature Selection via Kendall’s τ und Pearson-Korrelation

### Modelle
- **Random Survival Forest (RSF)** – für Survival-Analyse und Klassifikation
- **XGBoost AFT (Accelerated Failure Time)** – für RUL-Intervallvorhersage

### Evaluierung
- Kostenmatrix-basierte Entscheidung  
- Klassengrenzen (TAUs) → RUL-Klassen 0–4  
- Metriken: Accuracy, Balanced Accuracy, Expected Cost, Confusion Matrix

---

## 5. Tools & Libraries

- **Python 3.12**
- **tsfresh** – Feature-Engineering
- **scikit-survival**, **skranger** – Survival-Modelle
- **XGBoost** – AFT Modell
- **MLflow** – Experiment-Tracking
- **Streamlit** – Deployment UI
- **SHAP, Permutation Importance** – Explainability
- **OpenAI GPT-4o-mini** – Natürliche Sprach-Erklärungen der Modellentscheidungen

---

## 6. Experiment-Tracking mit MLflow

Alle Trainings- und Evaluierungsläufe werden mit **MLflow** protokolliert.  
Start der UI:

mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

7. Deployment
Das Projekt enthält eine interaktive Streamlit-App (src/app.py):

streamlit run /workspace/src/app.py

Features:

CSV-Upload eines einzelnen Readouts

Preprocessing-Pipeline (Interpolation, Differenzenbildung, Sliding Windows, tsfresh)

Modellvorhersage (RSF)

Visualisierung der Klassenscores (p₀–p₄)

Entscheidung basierend auf Kostenmatrix oder Argmax

GPT-gestützte Erklärungen der Modellentscheidung

## 8. Projektstruktur

```text

├── data/                 # Datenpipeline (01_raw – 08_reporting)
├── notebooks/            # Jupyter Notebooks (EDA, Modeling, Evaluation, Deployment)
├── src/                  # Python-Module (Preprocessing, Modelle, Streamlit-App)
│   ├── app.py            # Streamlit-App
│   ├── decision_utils.py # Entscheidungslogik (Kostenmatrix, Klassenwahrscheinlichkeit)
│   ├── preprocessing.py  # Datenaufbereitung
│   ├── local_feature_utils.py # Explainability (z.B. Feature Importance)
│   ├── openai_utils.py   # GPT-basierte Erklärungen
├── mlruns/               # MLflow Tracking Daten
├── env/.env              # API Keys (nicht versioniert)
├── Dockerfile            # Container-Build
├── docker-compose.yml    # Orchestrierung
├── README.md             # Projektdokumentation
├── requirements.txt      # Python Dependencies
└── streamlit_test_data.csv   # Test Data for Streamlit app
```

9. Reproduzierbarkeit & Container
Das gesamte Projekt kann mit Docker ausgeführt werden:
docker-compose up --build

10. Lizenz
Daten: CC BY 4.0 (SCANIA Component X Dataset, SND)

Code: MIT License (siehe Repository)
