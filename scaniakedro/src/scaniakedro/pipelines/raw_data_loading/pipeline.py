from kedro.pipeline import Pipeline, node
from .nodes import load_all_raw_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=load_all_raw_data,
            inputs=[
                "train_specifications",
                "train_operational_readouts",
                "train_tte",
                "validation_specifications",
                "validation_operational_readouts",
                "validation_labels",
                "test_specifications",
                "test_operational_readouts",
                "test_labels"
            ],
            outputs="raw_datasets",
            name="load_all_raw_data_node"
        )
    ])
