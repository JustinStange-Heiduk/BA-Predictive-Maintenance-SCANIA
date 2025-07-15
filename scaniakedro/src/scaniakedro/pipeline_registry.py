from __future__ import annotations

from kedro.pipeline import Pipeline
from scaniakedro.pipelines.raw_data_loading import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "raw_data_loading": create_pipeline(),
        "__default__": create_pipeline()
    }
