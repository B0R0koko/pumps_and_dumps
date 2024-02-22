from pathlib import Path

import pandas as pd
import os
import json


ROOT_DIR = Path(os.path.dirname(__file__)).parent
PIPE_PROGRESS_CFG = "features/pipes/cfg/pumps.json"
LABELED_PUMPS_PATH = "data/pumps/cleaned/pumps_verified.csv"
TRAIN_DATA_PATH = "features/data/train.parquet"


def reset_pipe_progress() -> None:
    """Reset pipe progress by resetting pipes/cfg/pumps.json and loads it with all pumps from data/pumps folder"""
    df_pumps = pd.read_csv(os.path.join(ROOT_DIR, LABELED_PUMPS_PATH))

    data = []

    for idx, row in df_pumps.iterrows():
        data.append({"ticker": row.ticker, "pump_time": row.time})

    with open(os.path.join(ROOT_DIR, PIPE_PROGRESS_CFG), "w") as file:
        json.dump(data, file)

    os.remove(os.path.join(ROOT_DIR, TRAIN_DATA_PATH))


if __name__ == "__main__":
    reset_pipe_progress()
