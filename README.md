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

## 2. Datenversionierung mit DVC

Um eine saubere, reproduzierbare Projektstruktur sicherzustellen, wird für das Datenmanagement DVC (Data Version Control) eingesetzt.

### Einrichtungsschritte:

1. Initialisierung von DVC (einmalig im Projektordner):
   
   dvc init
   git commit -m "Initialize DVC"

Hinzufügen des Rohdatenordners:

dvc add data/raw

Einbinden der Metadaten ins Git:

git add data/raw.dvc data/.gitignore
git commit -m "Track raw SCANIA data with DVC"