from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from datetime import timedelta, datetime
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from functools import partial
from tqdm import tqdm
from typing import *


import pandas as pd
import os
import re
import json


@dataclass
class PumpEvent:
    ticker: str
    time: str
    exchange: str

    def __post_init__(self):
        self.time: pd.Timestamp = pd.Timestamp(self.time)

    def __repr__(self):
        return f"{self.ticker}-{str(self.time)}-{self.exchange}"


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
            json.dump(data, file)


class DataLoader(ABC):
    """When inhereting from this class you need to implement preprocess and postprocess methods"""

    def __init__(
        self, 
        trades_dir: str = "data/trades_parquet",
        output_path: str = "data/datasets/train.parquet",
        cmc_snapshots_file: str = "data/cmc/cmc_snapshots.csv",
        labeled_pumps_file: str = "data/pumps/pumps_31_03_2024.json",
        lookback_period: timedelta = timedelta(days=30),
        warm_start: bool = False,
        progress_file: str = "feature_enj/progress.json",
        n_workers: int = 5,
        use_exchanges: List[str] = ["binance", "kucoin"],
        use_quotes: List[str] = ["BTC"]
    ) -> Self:
        
        self.trades_dir: str = trades_dir
        self.output_path: str = output_path
        self.cmc_snapshots_file: str = cmc_snapshots_file
        self.labeled_pumps_file: str = labeled_pumps_file
        # amount of time to lookback and calculate features on
        self.lookback_period: timedelta = lookback_period 
        self.warm_start: bool = warm_start
        self.progress_file: str = progress_file
        self.n_workers: int = n_workers
        self.use_exchanges: List[str] = use_exchanges
        self.use_quotes: List[str] = use_quotes


        self.available_timeframes: Dict[str, pd.DataFrame] = self.get_available_timeframes()
        self.pumps: List[PumpEvent] = self.load_pumps()
        self.df_cmc_snapshots: pd.DataFrame = self.load_cmc_snapshots()
        self.task_stack: TaskStack = TaskStack(
            pumps=self.pumps, progress_file=self.progress_file
        )

        if not self.warm_start:
            # Clear output_path
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def create_pumps(self, pumps: List[str]) -> List[PumpEvent]:
        """Wraps loaded json file into list of PumpEvent objects"""
        return [
            PumpEvent(
                ticker=pump["ticker"],
                time=pump["time"],
                exchange=pump["exchange"],
            )
            for pump in pumps if pump["exchange"] in self.use_exchanges
            # leave only pumps with selected exchanges
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
    
    def load_cmc_snapshots(self) -> pd.DataFrame:
        """
        Loads cmc snapshots into dataframe, this will be used for filtering by rank for each crosssection
        """
        df_cmc_snapshots: pd.DataFrame = pd.read_csv(self.cmc_snapshots_file)
        df_cmc_snapshots["date"] = pd.to_datetime(
            df_cmc_snapshots["snapshot"], format="%Y%m%d"
        )
        return df_cmc_snapshots
    
    def get_available_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        return {
            "binance": pd.DataFrame(
                ticker: str, available_from: pd.Timestamp, available_to: pd.Timestamp
            )
        }
        """
        map_exchange_df: Dict[str, pd.DataFrame] = {}
        for exchange in os.listdir(self.trades_dir):

            if exchange not in self.use_exchanges:
                continue

            exchange_dir: str = os.path.join(self.trades_dir, exchange)
            exchange_data = []

            for ticker in os.listdir(exchange_dir):
                # Make sure that we only store available timeframes for quotes that we actually want
                # to save RAM
                if not any([ticker.endswith(quote) for quote in self.use_quotes]):
                    continue

                ticker_files: List[str] = os.listdir(os.path.join(exchange_dir, ticker))

                available_dates: List[datetime] = [
                    datetime.strptime(
                        re.search(r"(\d{4}-\d{2}-\d{2})", file)[1], "%Y-%m-%d"
                    )
                    for file in ticker_files
                ]

                exchange_data.append({
                    "ticker": ticker,
                    "available_from": min(available_dates),
                    "available_to": max(available_dates),
                })

            map_exchange_df[exchange] = pd.DataFrame(exchange_data)

        return map_exchange_df
    
    
    def load_data(self, ticker: str, pump: PumpEvent) -> pd.DataFrame: 
        """Load data pd.DataFrame from trades_dir/exchange/ticker folder"""

        end: pd.Timestamp = pump.time.round("1h") - timedelta(hours=1)
        # data is organized by days
        date_range: List[pd.Timestamp] = pd.date_range(
            start=end - self.lookback_period, 
            end=end, 
            freq="D",
            inclusive="both"
        ).tolist()

        df: pd.DataFrame = pd.DataFrame()

        for date in date_range:
            file_name: str = f"{ticker}-trades-{date.date()}.parquet"
            file_path: str = os.path.join(self.trades_dir, pump.exchange, ticker, file_name)
            # Check if file exists
            if os.path.exists(file_path):
                df_date: pd.DataFrame = pd.read_parquet(
                    os.path.join(self.trades_dir, pump.exchange, ticker, file_name)
                )
                df_date["time"] = pd.to_datetime(df_date["time"], unit="ms")
                df_date["time"] = df_date["time"].dt.tz_localize(None)

                df = pd.concat([df, df_date])

        df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df[
            (df["time"] >= end - self.lookback_period) & (df["time"] <= end)
        ]

    @abstractmethod
    def preprocess_data(self, df_ticker: pd.DataFrame, ticker: str, pump: PumpEvent) -> pd.DataFrame:
        """
        this function is runned before create_features function. Allows to preprocess data
        loaded from local storage before feature creation
        """

    @abstractmethod
    def postprocess_data(self, df_features: pd.DataFrame, ticker: str, pump: PumpEvent) -> pd.DataFrame:
        """
        this function runs after create_features
        """

    @abstractmethod
    def get_crosssection_tickers(self, pump: PumpEvent) -> List[str]:
        """Get tickers that will used for crosssection for each pump"""

    @abstractmethod
    def create_features(
        self, df_ticker: pd.DataFrame, pump: PumpEvent, ticker: str
    ) -> pd.DataFrame:
        """When inherited we need to define this method that creates features for a given pump and ticker"""


    def create_crosssection(self, pump: PumpEvent) -> Tuple[PumpEvent, pd.DataFrame]:
        """
        Creates a crosssection within a period of time of the pump. Applies create_features method to each dataset and
        combines all transformed data into a single dataframe representing one crosssection.
        """
        df_crosssection: pd.DataFrame = pd.DataFrame()
        # Check if the pumped ticker is in collected data, otherwise no point in running code below
        tickers: List[str] | None = self.get_crosssection_tickers(pump=pump)

        if not tickers:
            return pump, df_crosssection
        
        tickers_iterable: List[str] | tqdm = tqdm(tickers, leave=False) if self.n_workers == 1 else tickers

        if self.n_workers == 1:
            tickers_iterable.set_description("Running in single process")

        for ticker in tickers_iterable:
            try:
                df_ticker: pd.DataFrame = self.load_data(ticker=ticker, pump=pump)
            except:
                continue
            # Apply preprocess before creating features
            df_ticker: pd.DataFrame = self.preprocess_data(df_ticker=df_ticker, ticker=ticker, pump=pump)
            df_features: pd.DataFrame = self.create_features(df_ticker=df_ticker, pump=pump, ticker=ticker)
            # Apply postprocess before creating features
            df_features: pd.DataFrame = self.postprocess_data(df_features=df_features, ticker=ticker, pump=pump)
            df_crosssection = pd.concat([df_crosssection, df_features])

        return pump, df_crosssection
    

    def transform(self) -> None:
        """Runs feature creation process in a single process mode"""
        pbar = tqdm(self.pumps)

        for pump in pbar:
            pbar.set_description(f"Creating crosssection for pump: {pump.ticker}")
            pump: PumpEvent
            df_crosssection: pd.DataFrame
            pump, df_crosssection = self.create_crosssection(pump=pump)

            if not df_crosssection.empty:
                df_crosssection.to_parquet(
                    self.output_path, compression="gzip", engine="fastparquet", 
                    append=os.path.exists(self.output_path), index=False
                )

            self.task_stack.pop(pump)


    def transform_multiprocess(self) -> None:
        """Runs feature creation with multiple processes"""
        with (
            tqdm(total=len(self.pumps), desc="Transforming in multiprocessing mode: ") as pbar,
            Pool(processes=self.n_workers) as pool,
        ):
            results = []

            for pump in self.pumps:
                res: AsyncResult = pool.apply_async(partial(self.create_crosssection, pump=pump))
                results.append(res)

            for res in results:
                # Once result is obtained increment pbar
                pump: PumpEvent
                df_crosssection: pd.DataFrame
                pump, df_crosssection = res.get()

                if not df_crosssection.empty:
                    df_crosssection.to_parquet(
                        self.output_path, compression="gzip", engine="fastparquet",
                        append=os.path.exists(self.output_path),
                    )

                self.task_stack.pop(pump)
                # update progress
                pbar.update(1)


    def run(self) -> None:
        if self.n_workers > 1:
            self.transform_multiprocess()
        else:
            self.transform()


def main():
    loader = DataLoader()

if __name__ == "__main__":
    main()