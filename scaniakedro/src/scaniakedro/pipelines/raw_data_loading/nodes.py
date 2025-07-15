import pandas as pd

def load_all_raw_data(
    train_specifications: pd.DataFrame,
    train_operational_readouts: pd.DataFrame,
    train_tte: pd.DataFrame,
    validation_specifications: pd.DataFrame,
    validation_operational_readouts: pd.DataFrame,
    validation_labels: pd.DataFrame,
    test_specifications: pd.DataFrame,
    test_operational_readouts: pd.DataFrame,
    test_labels: pd.DataFrame
) -> dict:
    result = {
        "train": {
            "spec": train_specifications,
            "readouts": train_operational_readouts,
            "tte": train_tte,
        },
        "validation": {
            "spec": validation_specifications,
            "readouts": validation_operational_readouts,
            "labels": validation_labels,
        },
        "test": {
            "spec": test_specifications,
            "readouts": test_operational_readouts,
            "labels": test_labels,
        }
    }
    return result.copy()  # <--- wichtig für MemoryDataset!
