from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from functools import partial
from tqdm import tqdm
from typing import *

import polars as pl
import pandas as pd
import json
import os
import re


@dataclass
class PumpEvent:
    ticker: str
    time: str
    exchange: str

    def __post_init__(self):
        self.time: pd.Timestamp = pd.Timestamp(self.time)

    def __str__(self):
        return f"Pump event: {self.ticker} - {str(self.time)} on {self.exchange}"

    def __repr__(self):
        return f"Pump event: {self.ticker} - {str(self.time)} on {self.exchange}"


class TaskStack:

    def __init__(self, pumps: List[PumpEvent], progress_file: str):
        # Kids, always remember to copy, I ran into some nasty bags with this
        self.pumps: List[PumpEvent] = pumps.copy()
        self.progress_file: str = progress_file

    def pop(self, pump: PumpEvent) -> None:
        """Handles updates of the progress.json file"""
        self.pumps.remove(pump)
        # save to config file

        data = []

        for pump in self.pumps:
            pump_dict = asdict(pump)
            pump_dict["time"] = str(pump_dict["time"])
            data.append(pump_dict)

        with open(os.path.join(self.progress_file), "w") as file:
            data = {"pumps": data}
            json.dump(data, file)


class DataLoader:

    def __init__(
        self,
        trades_dir: str = "data/trades_parquet",
        output_path: str = "data/datasets/train.parquet",
        cmc_snapshots_file: str = "data/cmc/cmc_snapshots.csv",
        labeled_pumps_file: str = "data/pumps/pumps_31_03_2024.json",
        cmc_rank_above: int = 100,
        warm_start: bool = False,
        progress_file: str = "features/progress.json",
        use_exchanges: List[str] = ["binance"],
    ):
        self.trades_dir: str = trades_dir
        self.output_path: str = output_path
        self.cmc_snapshots_file: str = cmc_snapshots_file
        self.labeled_pumps_file: str = labeled_pumps_file
        self.cmc_rank_above: int = cmc_rank_above
        self.warm_start: bool = warm_start
        self.progress_file: str = progress_file
        self.use_exchanges: List[str] = use_exchanges

        self.tickers_collected: Dict[str, List[str]] = self.load_tickers()
        self.df_cmc_snapshots: pd.DataFrame = self.load_cmc_snapshots()

        self.available_timeframes: Dict[str, pd.DataFrame] = (
            self.get_tickers_available_timeframes()
        )
        self.pumps: List[PumpEvent] = self.load_pumps()

        self.task_stack: TaskStack = TaskStack(
            pumps=self.pumps, progress_file=self.progress_file
        )

    def load_tickers(self) -> Dict[str, List[str]]:
        """Load all tickers collected for a given exchange"""

        """
        Function returns the following dict
        {
            "binance": ["ADABTC", "EZBTC" ....],
            "kucoin": ["AVAXUSDT", ...],
            "MEXC": [...]
        }
        """
        # each folder in TRADES_DIR must be named by the corresponding exchange name
        self.available_exchanges: List[str] = os.listdir(self.trades_dir)

        tickers_collected: Dict[str, List[str]] = dict()

        for exchange in self.available_exchanges:
            tickers_collected[exchange] = os.listdir(
                os.path.join(self.trades_dir, exchange)
            )

        return tickers_collected

    def load_cmc_snapshots(self) -> pd.DataFrame:
        """Loads cmc snapshots into dataframe, this will be used for filtering by rank for each crosssection"""
        df_cmc_snapshots: pd.DataFrame = pd.read_csv(self.cmc_snapshots_file)
        df_cmc_snapshots["snapshot"] = pd.to_datetime(
            df_cmc_snapshots["snapshot"], format="%Y%m%d"
        )
        return df_cmc_snapshots

    @staticmethod
    def create_pumps(pumps: List[str]) -> List[PumpEvent]:
        """Wraps loaded json file into list of PumpEvent objects"""
        return [
            PumpEvent(
                ticker=pump["ticker"],
                time=pump["time"],
                exchange=pump["exchange"],
            )
            for pump in pumps
        ]

    def load_pumps(self) -> List[PumpEvent]:
        """
        if self.warm_start is true -> Load pumps that were not preprocessed by DataLoader
        from progress.json file, itherwise load all pumps and update progress.json with all pumps
        """
        if self.warm_start:
            # If we want to run from warm start
            with open(self.progress_file, "r") as file:
                pumps: List[Dict[str, Any]] = json.load(file)
            return self.create_pumps(pumps=pumps)

        # otherwise read all pumps and populate progress.json stack with them
        with open(self.labeled_pumps_file, "r") as file:
            pumps: List[str] = json.load(file)
        # Populate progress.json
        with open(self.progress_file, "w") as file:
            json.dump(pumps, file)

        return self.create_pumps(pumps=pumps)

    def get_tickers_available_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        return {
            "binance": pd.DataFrame(
                ticker: str, available_from: pd.Timestamp, available_to: pd.Timestamp
            )
        }
        """
        map_exchange_df: Dict[str, pd.DataFrame] = {}

        for exchange in os.listdir(self.trades_dir):
            exchange_dir: str = os.path.join(self.trades_dir, exchange)

            exchange_data = []

            for ticker in os.listdir(exchange_dir):
                ticker_files: List[str] = os.listdir(os.path.join(exchange_dir, ticker))

                available_dates: List[datetime] = [
                    datetime.strptime(
                        re.search(r"(\d{4}-\d{2}-\d{2})", file)[1], "%Y-%m-%d"
                    )
                    for file in ticker_files
                ]

                exchange_data.append(
                    {
                        "ticker": ticker,
                        "available_from": min(available_dates),
                        "available_to": max(available_dates),
                    }
                )

            map_exchange_df[exchange] = pd.DataFrame(exchange_data)

        return map_exchange_df

    def get_tickers_for_crosssection(
        self, pump: PumpEvent, lookback_delta: timedelta
    ) -> List[str]:
        """
        For each pump we create a crosssection, we would like to create a crosssection using top-50 + tickers,
        otherwise we have to calculate metrics for tickers that are traded a lot => a lot of computation which is useless
        since such tickers will never be pumped in the first place
        """
        df_cmc: pd.DataFrame = self.df_cmc_snapshots.copy()
        df_cmc["time_diff"] = (df_cmc["snapshot"] - pump.time).abs()  # abs timedelta

        shit_coins: List[str] = df_cmc[
            (df_cmc["time_diff"] == df_cmc["time_diff"].min())
            & (df_cmc["cmc_rank"] > self.cmc_rank_above)
        ]["symbol"].tolist()

        shit_coins = [f"{el}BTC" for el in shit_coins]

        shit_coins: set = set(shit_coins) | set([pump.ticker])
        # create a set of tickers traded on Pump.exchange within timeframe
        # [Pump.time - lookback_delta, Pump.time]

        df_timeframes: pd.DataFrame = self.available_timeframes[pump.exchange]

        HAS_ENOUGH_DATA = (
            df_timeframes["available_from"] <= (pump.time - lookback_delta)
        ) & (pump.time <= df_timeframes["available_to"])

        collected_tickers: set = set(df_timeframes[HAS_ENOUGH_DATA]["ticker"].tolist())
        # intersect these tickers with ones form the shit_coins
        tickers: List[str] = list(
            # also include pumped ticker
            shit_coins.intersection(collected_tickers)
        )

        return tickers

    def load_data(
        self, ticker: str, pump: PumpEvent, lookback_delta: timedelta
    ) -> pd.DataFrame:
        """Load data pd.DataFrame from trades_dir/exchange/ticker folder"""

        # data is organized by days
        date_range: List[pd.Timestamp] = pd.date_range(
            start=pump.time.round("1h") - lookback_delta - timedelta(hours=1),
            end=pump.time.round("1h") - timedelta(hours=1),
            freq="D",
        ).tolist()

        df: pd.DataFrame = pd.DataFrame()

        for date in date_range:
            file_name: str = f"{ticker}-trades-{date.date()}.parquet"
            try:
                df_date: pd.DataFrame = pd.read_parquet(
                    os.path.join(self.trades_dir, pump.exchange, ticker, file_name)
                )
                df = pd.concat([df, df_date])
            except:
                pass

        return df[
            (df["time"] >= pump.time.round("1h") - lookback_delta - timedelta(hours=1))
            & (df["time"] <= pump.time.round("1h") - timedelta(hours=1))
        ]

    def create_crosssection(
        self, pump: PumpEvent, lookback_delta: timedelta
    ) -> Tuple[PumpEvent, pd.DataFrame]:
        """
        Creates a crosssection within a period of time of the pump. Applies create_features method to each dataset and
        combines all transformed data into a single dataframe representing one crosssection.
        """
        df_crosssection: pd.DataFrame = pd.DataFrame()
        # Check if the pumped ticker is in collected data, otherwise no point in running code below

        tickers: List[str] = self.get_tickers_for_crosssection(
            pump=pump, lookback_delta=lookback_delta
        )

        if pump.ticker not in tickers:
            return pump, df_crosssection

        for ticker in tickers:
            df_ticker: pd.DataFrame = self.load_data(
                ticker=ticker, pump=pump, lookback_delta=lookback_delta
            )
            df_features: pd.DataFrame = self.create_features(
                df_ticker=df_ticker, pump=pump
            )
            df_features["ticker"] = ticker
            df_features["pumped_ticker"] = pump.ticker
            df_features["pump_time"] = str(pump.time)

            df_crosssection = pd.concat([df_crosssection, df_features])

        return pump, df_crosssection

    @abstractmethod
    def create_features(self, df_ticker: pl.DataFrame, pump: PumpEvent) -> pd.DataFrame:
        """When inherited we need to define this method that creates features for a given pump and ticker"""

    def run(self) -> None:
        """Runs feature creation process in a single process mode"""
        pbar = tqdm(self.pumps)

        for pump in pbar:
            pbar.set_description(f"Creating crosssection for pump: {pump.ticker}")

            if pump.exchange not in self.use_exchanges:
                continue

            df_crosssection: pd.DataFrame = self.create_crosssection(
                pump=pump, lookback_delta=timedelta(days=30)
            )

    def run_multiprocess(self) -> None:
        """Runs feature creation with multiple processes"""
        with (
            tqdm(
                total=len(self.pumps),
                desc="Transforming in multiprocessing mode: ",
            ) as pbar,
            Pool(processes=10) as pool,
        ):
            results = []

            for pump in self.pumps:
                res: AsyncResult = pool.apply_async(
                    partial(
                        self.create_crosssection,
                        pump=pump,
                        lookback_delta=timedelta(days=30),
                    )
                )
                results.append(res)

            for res in results:
                # Once result is obtained increment pbar
                pump, df_crosssection = res.get()

                try:
                    df_crosssection.to_parquet(
                        self.output_path,
                        compression="gzip",
                        engine="fastparquet",
                        append=os.path.exists(self.output_path),
                    )

                except ValueError:
                    pass

                finally:
                    self.task_stack.pop(pump)
                    # update progress
                    pbar.update(1)


if __name__ == "__main__":
    loader = DataLoader(
        trades_dir="data/trades_parquet",
        output_path="data/datasets/train.parquet",
        cmc_snapshots_file="data/cmc/cmc_snapshots.csv",
        labeled_pumps_file="data/pumps/pumps_31_03_2024.json",
        cmc_rank_above=100,
        warm_start=False,
        progress_file="features/progress.json",
    )
