from pathlib import Path
from typing import *
from feature_pipes.transform import DataLoader, PumpEvent
from feature_pipes.transform_features import transform_features

import numpy as np
import pandas as pd
import polars as pl
import os
import warnings
import cProfile


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def create_features(
        self, pump_event: PumpEvent, df_ticker: pl.DataFrame
    ) -> pd.DataFrame:
        # Perform all feature engineering here
        # Leave only the last 24 hours of the data before pump
        try:
            df_features: pd.DataFrame = transform_features(
                df=df_ticker, pump_event=pump_event, window="7d"
            )
            return df_features
        except:
            return pd.DataFrame()


if __name__ == "__main__":

    import cProfile
    import pstats

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

    loader = Loader(
        data_dir="data/trades",
        labeled_pumps_file="data/pumps/cleaned/pumps_verified.csv",
    )

    with cProfile.Profile() as pr:
        df_train: pd.DataFrame = loader.multiprocess_transform_data()
        df_train.to_parquet(os.path.join(ROOT_DIR, "data/data_transformed.parquet"))

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling.prof")
