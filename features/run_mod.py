from typing import List
from pipes.dataloader_mod import DataLoader, PumpEvent
from pipes.features import transform_to_simple_features
from datetime import timedelta

import pandas as pd
import re
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def get_crosssection_tickers(self, pump: PumpEvent) -> List[str]:
        
        CMC_RANK_ABOVE = 100
        
        """
        For each pump we create a crosssection, we would like to create a crosssection using top-50 + tickers,
        otherwise we have to calculate metrics for tickers that are traded a lot => a lot of computation which is useless
        since such tickers will never be pumped in the first place
        """
        self.df_cmc_snapshots["time_diff"] = (
            self.df_cmc_snapshots["snapshot"] - pump.time
        ).abs()  # abs timedelta

        shit_coins: List[str] = self.df_cmc_snapshots[
            (self.df_cmc_snapshots["time_diff"] == self.df_cmc_snapshots["time_diff"].min())
            & (self.df_cmc_snapshots["cmc_rank"] >= CMC_RANK_ABOVE)
        ]["symbol"].tolist()

        quote_asset: str = "BTC" if pump.exchange == "binance" else "USDT"
        shit_coins: List[str] = [f"{el}{quote_asset}" for el in shit_coins]

        shit_coins: set = set(shit_coins) | set([pump.ticker])
        # create a set of tickers traded on pump.exchange within timeframe
        # [pump.time - lookback_period, pump.time]

        df_timeframes: pd.DataFrame = self.available_timeframes[pump.exchange]

        HAS_ENOUGH_DATA = (
            df_timeframes["available_from"] <= (pump.time - self.lookback_period)
        ) & (pump.time <= df_timeframes["available_to"])

        collected_tickers: set = set(df_timeframes[HAS_ENOUGH_DATA]["ticker"].tolist())
        # intersect these tickers with ones form the shit_coins
        tickers: List[str] = list(
            # also include pumped ticker
            shit_coins.intersection(collected_tickers)
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
        base_asset: str = re.sub(r"(BTC|USDT)$", "", pump.ticker)

        df_cmc_ticker: pd.DataFrame = self.df_cmc_snapshots[
            (self.df_cmc_snapshots["symbol"] == base_asset) &
            (self.df_cmc_snapshots["snapshot"] < pump.time.floor("1d"))
        ].copy()

        df_ticker_features: pd.DataFrame = transform_to_simple_features(
            df_ticker=df_ticker, pump=pump, df_cmc_ticker=df_cmc_ticker, ticker=ticker
        )

        return df_ticker_features


if __name__ == "__main__":
    loader = Loader(
        trades_dir="data/trades_parquet",
        output_path="data/datasets/train_16_04.parquet",
        cmc_snapshots_file="data/cmc/cmc_snapshots.csv",
        labeled_pumps_file="data/pumps/pumps_31_03_2024.json",
        lookback_period=timedelta(days=30),
        warm_start=False,
        progress_file="features/progress.json",
        n_workers=5,
        use_exchanges=["binance"]
    )
    loader.run()
