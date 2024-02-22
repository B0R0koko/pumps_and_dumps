from pipes.dataloader import DataLoader, PumpEvent
from pipes.features import transform_features


import numpy as np
import pandas as pd
import polars as pl
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def create_features(
        self, pump_event: PumpEvent, df_ticker: pl.DataFrame
    ) -> pd.DataFrame:
        # Perform all feature engineering here
        # Leave only the last 24 hours of the data before pump
        df_features: pd.DataFrame = transform_features(
            df=df_ticker, pump_event=pump_event, window="7d", window_output="1h"
        )
        return df_features


if __name__ == "__main__":
    loader = Loader(data_dir="data/trades", output_path="features/data/train.parquet")
    loader.run()
