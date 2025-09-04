# SCANIA Component X – Predictive Maintenance Projekt

## 1. Datenquelle

Die für dieses Projekt verwendeten Daten stammen aus dem öffentlich zugänglichen Forschungskatalog des **Swedish National Data Service (SND)**:

> **Titel:** SCANIA Component X Dataset  
> **Quelle:** https://researchdata.se/en/catalogue/dataset/2024-34/2  
> **Zugriffsdatum:** 14. Juli 2025  
> **Lizenz:** CC BY 4.0 (Creative Commons Attribution)

Das Dataset umfasst neun separate CSV-Dateien für Training, Validierung und Testdaten. Enthalten sind:
- Spezifikationen der Komponenten
- Zeitreihen von Betriebsmesswerten (Operational Readouts)
- Labels bzw. Restlebensdauer (TTE = time-to-event)

Die Dateien wurden nach dem Download unverändert gespeichert unter:

data/raw/
├── train_specifications.csv
├── train_operational_readouts.csv
├── train_tte.csv
├── validation_specifications.csv
├── validation_operational_readouts.csv
├── validation_labels.csv
├── test_specifications.csv
├── test_operational_readouts.csv
├── test_labels.csv

---


# 2. Data Ordner Struktur

Ordner	Zweck	Beispiele aus deinem Projekt
01_raw	Unveränderte Rohdaten, keine Bearbeitung	Original component_X_train.csv, readouts.csv
02_intermediate	Vorverarbeitete Daten (Diff, Interpolation, dropna)
03_primary	Integrierte, bereinigte Struktur (Window)
04_feature	Von tsfresh o. Ä. generierte Features oder manuell berechnete Features	features_window8.csv, features_window32.csv
05_model_input	Finalisierte Inputdaten für das Modell – nach Feature Selection und evtl. Skalierung	X_train.csv, y_train.csv, X_val.csv
06_models	Gespeicherte Modelle	xgboost_model.pkl, bilstm_model.h5
07_model_output	Modelloutputs wie Vorhersagen, Wahrscheinlichkeiten, Residuen	y_pred.csv, rul_pred.csv, SHAP-Werte
08_reporting	Abbildungen, Tabellen, Metriken, Explainability, etc.	ROC-Curve, Feature Importance, Evaluation Reports


mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
