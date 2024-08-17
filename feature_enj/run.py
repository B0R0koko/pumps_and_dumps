from typing import List
from pipes.dataloader import DataLoader, PumpEvent
from datetime import timedelta
from features.features_17_08.compute import transform_to_features
import pandas as pd
import re
import os
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def get_crosssection_tickers(self, pump: PumpEvent) -> List[str] | None:

        TOP_CMC_RANK = 100

        """
        For each pump we create a crosssection, we would like to create a crosssection using top-50 + tickers,
        otherwise we have to calculate metrics for tickers that are traded a lot => a lot of computation which is useless
        since such tickers will never be pumped in the first place
        """
        self.df_cmc_snapshots["time_diff"] = (self.df_cmc_snapshots["date"] - pump.time).abs()  # abs timedelta

        top_coins: List[str] = self.df_cmc_snapshots[
            (self.df_cmc_snapshots["time_diff"] == self.df_cmc_snapshots["time_diff"].min())
            & (self.df_cmc_snapshots["cmc_rank"] <= TOP_CMC_RANK)
        ]["symbol"].tolist()

        quote_asset: str = "BTC" if pump.exchange == "binance" else "USDT"
        top_coins: List[str] = set([f"{el}{quote_asset}" for el in top_coins])

        # create a set of tickers traded on pump.exchange within timeframe
        # [pump.time - lookback_period, pump.time]

        df_timeframes: pd.DataFrame = self.available_timeframes[pump.exchange]

        HAS_ENOUGH_DATA: pd.Series[bool] = (df_timeframes["available_from"] <= (pump.time - self.lookback_period)) & (
            pump.time <= df_timeframes["available_to"]
        )

        collected_tickers: List[str] = df_timeframes[HAS_ENOUGH_DATA]["ticker"].tolist()
        collected_tickers: set = set([ticker for ticker in collected_tickers if ticker.endswith("BTC")])

        if pump.ticker not in collected_tickers:
            # if pumped ticker doesn't have enough data return no tickers
            return
        # intersect these tickers with ones form the shit_coins
        tickers: List[str] = list(
            (collected_tickers - top_coins)
            | set([pump.ticker])  # remove top tickers and union add pumped ticker which has enough data
        )

        return tickers

    def preprocess_data(self, df_ticker: pd.DataFrame, ticker: str, pump: PumpEvent) -> pd.DataFrame:

        if pump.exchange == "kucoin":
            # if buyer is maker, then someone else matched his order => SELL
            df_ticker["isBuyerMaker"] = df_ticker["side"] == "SELL"
            df_ticker = df_ticker.drop(columns=["side"])

        return df_ticker

    def postprocess_data(self, df_features: pd.DataFrame, ticker: str, pump: PumpEvent) -> pd.DataFrame:
        # Add features on top of ones created in create_features
        df_features["exchange"] = pump.exchange
        df_features["pumped_ticker"] = pump.ticker
        df_features["pump_time"] = str(pump.time)
        df_features["ticker"] = ticker
        df_features["is_pumped"] = ticker == pump.ticker
        # add feature measuring number of days the ticker has been listed on binance,
        df_exchange: pd.DataFrame = self.available_timeframes[pump.exchange]

        available_from: pd.Timestamp = df_exchange[df_exchange["ticker"] == ticker]["available_from"].iloc[0]
        days_listed: int = (pump.time - available_from).days
        df_features["days_listed"] = days_listed

        return df_features

    def create_features(self, pump: PumpEvent, df_ticker: pd.DataFrame, ticker: str) -> pd.DataFrame:
        # Perform all feature engineering here
        base_asset: str = re.sub(r"(BTC)$", "", ticker)

        df_cmc_ticker: pd.DataFrame = self.df_cmc_snapshots[
            (self.df_cmc_snapshots["symbol"] == base_asset)
            & (self.df_cmc_snapshots["date"] < pump.time.floor("1d"))
            & (self.df_cmc_snapshots["date"] >= pump.time.floor("1d") - self.lookback_period)
        ].copy()

        try:
            df_ticker_features: pd.DataFrame = transform_to_features(
                df_ticker=df_ticker, pump=pump, df_cmc_ticker=df_cmc_ticker, ticker=ticker
            )
            return df_ticker_features
        except:
            return pd.DataFrame()


if __name__ == "__main__":

    loader = Loader(
        trades_dir="data/trades_parquet",
        output_path="data/datasets/train_17_08.parquet",
        cmc_snapshots_file="data/cmc/cmc_snapshots.csv",
        labeled_pumps_file="data/pumps/merged_pumps_15_08.json",
        lookback_period=timedelta(days=30),
        warm_start=False,
        progress_file="feature_enj/progress.json",
        n_workers=10,
        use_exchanges=["binance"],
        use_quotes=["BTC"],
    )

    loader.run()
