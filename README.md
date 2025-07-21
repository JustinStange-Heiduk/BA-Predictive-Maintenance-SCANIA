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


## 3. Projektstruktur und Workflow mit Kedro

Für die Organisation, Modularisierung und Reproduzierbarkeit des Machine-Learning-Projekts wird [Kedro](https://kedro.org/) verwendet.

### Initialisierung eines Kedro-Projekts

```bash
kedro new
# Folge dem Wizard und gib z.B. "scaniakedro" als Projektnamen an
```

### Wichtige Ordner und Dateien

```
scaniakedro/
├── conf/
│   ├── base/
│   │   ├── catalog.yml        # Definition aller Datasets
│   │   ├── parameters.yml     # Globale Parameter
│   │   └── logging.yml        # Logging-Konfiguration
│   └── local/                 # Lokale, nicht versionierte Einstellungen
├── data/
│   ├── 01_raw/                # Rohdaten (von DVC verwaltet)
│   ├── 02_intermediate/       # Zwischenergebnisse
│   ├── 03_primary/            # Modellierbare Daten
│   └── 04_model_input/        # Daten für Modelltraining
├── src/
│   └── scaniakedro/
│       ├── pipelines/         # Alle Pipelines (z.B. raw_data_loading)
│       │   └── raw_data_loading/
│       │       ├── nodes.py   # Funktionen für die Pipeline
│       │       └── pipeline.py# Pipeline-Definition
│       ├── __init__.py
│       └── pipeline_registry.py # Pipeline-Register
├── notebooks/                 # Jupyter Notebooks für Exploration
├── requirements.txt           # Paketabhängigkeiten
└── README.md
```

### Typischer Workflow mit Kedro

1. **Projekt konfigurieren:**
   ```python
   from kedro.framework.project import configure_project
   configure_project("scaniakedro")
   ```

2. **Session starten und Pipeline ausführen:**
   ```python
   from kedro.framework.session import KedroSession
   with KedroSession.create() as session:
       context = session.load_context()
       result = session.run(pipeline_name="raw_data_loading")
   ```

3. **Datenkatalog nutzen:**  
   Die Datasets werden in `conf/base/catalog.yml` definiert und können in Nodes und Pipelines verwendet werden.

4. **Eigene Pipelines und Nodes:**  
   Neue Pipelines werden im Ordner `src/scaniakedro/pipelines/` angelegt und im `pipeline_registry.py` registriert.

### Vorteile von Kedro

- **Modularität:** Klare Trennung von Daten, Code und Konfiguration.
- **Reproduzierbarkeit:** Jeder Schritt ist versioniert und nachvollziehbar.
- **Automatisiertes Datenhandling:** Datasets werden zentral im Data Catalog verwaltet.
- **Pipeline-Management:** Komplexe Workflows sind einfach abbildbar und ausführbar.

---

Weitere Infos:  
- [Kedro Dokumentation](https://docs.kedro.org/)
- [DVC Dokumentation](https://dvc.org/doc)