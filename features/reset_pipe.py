from typing import *

import pandas as pd
import json


LABELED_PUMPS_FILE = "data/pumps/pumps_collected_31_03_2024.csv"
PROGRESS_FILE = "features/progress.json"


def reset_pipe_progress() -> None:
    """Reset progress.json by resetting to all pumps in LABELED_PUMPS_FILE"""
    df_pumps: pd.DataFrame = pd.read_csv(LABELED_PUMPS_FILE)
    # represent dataframe as a list of dicts
    data: List[Dict[str, Any]] = df_pumps.to_dict(orient="records")

    with open(PROGRESS_FILE, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    reset_pipe_progress()
