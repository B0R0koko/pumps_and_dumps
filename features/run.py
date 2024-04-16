from pipes.dataloader import DataLoader, PumpEvent
from pipes.features import transform_to_simple_features

import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def create_features(self, pump: PumpEvent, df_ticker: pd.DataFrame) -> pd.DataFrame:
        # Perform all feature engineering here
        # Leave only the last 24 hours of the data before pump
        df_cmc_ticker: pd.DataFrame = self.df_cmc_snapshots[
            self.df_cmc_snapshots["symbol"] == f"{pump.ticker.replace("BTC", "")}"
        ].copy()

        df_ticker_features: pd.DataFrame = transform_to_simple_features(
            df_ticker=df_ticker, pump=pump, df_cmc_ticker=df_cmc_ticker
        )
        return df_ticker_features


if __name__ == "__main__":
    loader = Loader(
        trades_dir="data/trades_parquet",
        output_path="data/datasets/train_12_04.parquet",
        cmc_snapshots_file="data/cmc/cmc_snapshots.csv",
        labeled_pumps_file="data/pumps/pumps_31_03_2024.json",
        cmc_rank_above=100,
        warm_start=False,
        progress_file="features/progress.json",
        use_exchanges=["binance"],
    )
    loader.run_multiprocess()
