import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MultiprocessingDistributor

# -------------------------
# Interpolation
# -------------------------
def interpolate_readout_df(df: pd.DataFrame) -> pd.DataFrame:
    """Interpoliert fehlende Sensordatenwerte pro Fahrzeug linear.

    Args:
        df (pd.DataFrame): Eingabedaten mit Spalten
            ['vehicle_id', 'time_step', <Sensorspalten>].

    Returns:
        pd.DataFrame: Datenframe mit interpolierten Sensordaten,
        sortiert nach Fahrzeug und Zeit, Index zurückgesetzt.
    """
    df = df.sort_values(by=["vehicle_id", "time_step"])
    feature_cols = df.select_dtypes(include=["number"]).columns.difference(["vehicle_id", "time_step"])
    df[feature_cols] = df.groupby("vehicle_id")[feature_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    return df.reset_index(drop=True)


# -------------------------
# Differenzen berechnen
# -------------------------
def compute_differences_per_vehicle_test(readouts_df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet zeitliche Differenzen pro Fahrzeug für alle Sensormesswerte.

    Args:
        readouts_df (pd.DataFrame): Eingabedaten mit Spalten
            ['vehicle_id', 'time_step', <Sensorspalten>].

    Returns:
        pd.DataFrame: Ursprüngliche Daten mit zusätzlichen Spalten
        <sensor>_diff, welche die Differenzen darstellen.
    """
    df = readouts_df.copy()
    bin_cols = [col for col in df.columns.difference(["vehicle_id", "time_step"])]
    for col in bin_cols:
        new_col = f"{col}_diff"
        df[new_col] = df.groupby("vehicle_id")[col].diff()
        first_values = df.groupby("vehicle_id")[col].transform("first")
        df[new_col] = df[new_col].fillna(first_values)
    return df


# -------------------------
# Windowing
# -------------------------
def create_all_fixed_time_index_windows(readouts_df: pd.DataFrame, window_sizes: list[float]) -> pd.DataFrame:
    """Erzeugt Sliding Windows über Zeitreihen für jedes Fahrzeug.

    Args:
        readouts_df (pd.DataFrame): Eingabedaten mit Spalten
            ['vehicle_id', 'time_step', <Sensorspalten>].
        window_sizes (list[float]): Liste der Fenstergrößen (z. B. [8, 16, 32]).

    Returns:
        pd.DataFrame: Umgeformte Zeitfenster in Long-Format mit Spalten
        ['id', 'vehicle_id', 'time_step', 'time_step_current', 'kind', 'value'].
    """
    base_cols = ["vehicle_id", "time_step"]
    sensor_cols = [c for c in readouts_df.columns if c not in base_cols]

    melted = readouts_df.melt(
        id_vars=base_cols,
        value_vars=sensor_cols,
        var_name="kind",
        value_name="value"
    )

    melted = melted.sort_values(["vehicle_id", "time_step"], kind="mergesort", ignore_index=True)
    out_parts: list[pd.DataFrame] = []

    for w in window_sizes:
        for vid, g in melted.groupby("vehicle_id"):
            t = g["time_step"].to_numpy()
            uniq_t = np.unique(t)
            starts = np.searchsorted(t, uniq_t - w, side="right")
            ends   = np.searchsorted(t, uniq_t, side="left")

            added = False  # Track, ob überhaupt ein Fenster erzeugt wurde

            for i_start, i_end, ct in zip(starts, ends, uniq_t):
                if i_end <= i_start:
                    continue
                win_slice = g.iloc[i_start:i_end].copy()
                win_slice["id"] = f"vid{vid}_t{ct}_w{w}"
                win_slice["time_step_current"] = np.float64(ct)
                out_parts.append(win_slice)
                added = True

            # Fallback: nur ein Readout – trotzdem aufnehmen
            if not added and len(g["time_step"].unique()) == 1:
                ct = g["time_step"].iloc[0]
                win_slice = g.copy()
                win_slice["id"] = f"vid{vid}_t{ct}_w{w}_fallback"
                win_slice["time_step_current"] = np.float64(ct)
                out_parts.append(win_slice)

    if not out_parts:
        print("⚠️ Keine Fenster erzeugt – auch kein Fallback möglich.")
        return pd.DataFrame()

    final = pd.concat(out_parts, ignore_index=True)
    return final[["id", "vehicle_id", "time_step", "time_step_current", "kind", "value"]]


# -------------------------
# Feature Extraction
# -------------------------
def extract_tsfresh_features(df_windows: pd.DataFrame, n_workers: int = 4) -> pd.DataFrame:
    """Extrahiert tsfresh-Features aus Sliding Windows.

    Args:
        df_windows (pd.DataFrame): Zeitfenster im Long-Format mit Spalten
            ['id', 'vehicle_id', 'time_step', 'time_step_current', 'kind', 'value'].
        n_workers (int, optional): Anzahl paralleler Prozesse. Defaults to 4.

    Returns:
        pd.DataFrame: Feature-Datenframe mit berechneten tsfresh-Merkmalen,
        inkl. Meta-Infos ['vehicle_id', 'time_step'].
    """
    distributor = MultiprocessingDistributor(
        n_workers=n_workers,
        disable_progressbar=True
    )
    
    selected_fc_parameters = {
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "minimum": None,
        "maximum": None,
    }

    features_df = extract_features(
        df_windows,
        column_id="id",
        column_sort="time_step",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=selected_fc_parameters,
        impute_function=impute,
        distributor=distributor
    )

    id_metadata = df_windows[["id", "vehicle_id", "time_step_current"]].drop_duplicates("id")
    features_df = features_df.merge(id_metadata, how="left", left_index=True, right_on="id")
    features_df = features_df.set_index("id")
    features_df = features_df.rename(columns={"time_step_current": "time_step"})
    return features_df


# -------------------------
# Feature Selection
# -------------------------
def select_relevant_features(df_features: pd.DataFrame, selected_features: dict) -> pd.DataFrame:
    """Filtert relevante Features anhand einer übergebenen Feature-Liste.

    Args:
        df_features (pd.DataFrame): Vollständiger tsfresh Feature-Datenframe.
        selected_features (dict): Mapping relevanter Features (Feature-Name → p-Wert).

    Returns:
        pd.DataFrame: Gefilterte Feature-Matrix mit nur relevanten Spalten.
    """
    feature_cols = list(selected_features.keys())
    return df_features[feature_cols]
