from functools import partial
from multiprocessing.pool import AsyncResult
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing import Pool, freeze_support, RLock, current_process
from datetime import datetime, timedelta
from typing import *

import polars as pl
import pandas as pd
import os
import re


@dataclass
class PumpEvent:
    ticker: str
    pump_time: str

    def __post_init__(self):
        self.pump_time = datetime.strptime(self.pump_time, "%Y-%m-%d %H:%M:%S")

    def __str__(self) -> str:
        return f"{self.ticker}_{self.pump_time.date()}_{str(self.pump_time.time()).replace(":", "-")}"


@dataclass
class DataTickerTimeFrame:
    ticker: str
    available_start_date: datetime
    available_end_date: datetime


class DataLoader(ABC):
    """
    Parent class to handle feature creation. Inherit from this class and define abstract create_features method
    """

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

    def __init__(self, data_dir: str, labeled_pumps_file: str) -> None:
        self.data_dir = os.path.join(self.ROOT_DIR, data_dir)
        self.labeled_pumps_file = os.path.join(self.ROOT_DIR, labeled_pumps_file)

        self.available_timeframes: List[DataTickerTimeFrame] = (
            self.map_existing_trade_data_to_time_intervals()
        )

        self.labeled_pumps: List[PumpEvent] = self.load_pumps()

    def load_pumps(self) -> List[PumpEvent]:
        df_pumps: pl.DataFrame = pl.read_csv(self.labeled_pumps_file)
        return [
            PumpEvent(ticker=pump[0],pump_time=pump[1]) for pump in df_pumps.iter_rows()
        ]

    def map_existing_trade_data_to_time_intervals(
        self,
    ) -> Dict[str, DataTickerTimeFrame]:
        """Returns a list of all tickers collected and stored in self.data_dir directory.
        It defines start_month and end_month of data available for each ticker. This is needed when taking a crosssection
        of the pump."""
        available_timeframes: Dict[str, DataTickerTimeFrame] = {}

        DATA_DIR = os.path.join(self.ROOT_DIR, self.data_dir)

        for folder in os.listdir(DATA_DIR):
            # folder is the name of the ticker
            TICKER_DIR = os.path.join(DATA_DIR, folder)
            parquet_files: List[str] = os.listdir(TICKER_DIR)

            available_dates: List[datetime] = [
                datetime.strptime(re.search(r"-(\d{4}-\d{2}).parquet", file)[1], "%Y-%m")
                for file in parquet_files
            ]

            start_date, end_date = min(available_dates), max(available_dates)

            available_timeframes[folder] = DataTickerTimeFrame(
                ticker=folder,
                available_start_date=start_date,
                available_end_date=end_date,
            )

        return available_timeframes

    @staticmethod
    def create_date_range(start: datetime, end: datetime) -> List[datetime]:
        """Creates a range of months and years between two dates"""
        start_year = start.year
        start_month = start.month
        end_year = end.year
        end_month = end.month

        date_range = []

        for year in range(start_year, end_year + 1):
            start_month_range = start_month if year == start_year else 1
            end_month_range = end_month if year == end_year else 12

            for month in range(start_month_range, end_month_range + 1):
                date_range.append(datetime(year=year, month=month, day=1))

        return date_range
    
     
    def read_data(self, ticker: str, pump_event: PumpEvent, lookback_delta: timedelta) -> pl.DataFrame:
        rb = pump_event.pump_time
        lb = rb - lookback_delta
   
        ts_range = self.create_date_range(start=lb, end=rb)
        ticker_tf: DataTickerTimeFrame = self.available_timeframes[ticker]

        if not (ticker_tf.available_start_date <= lb <= rb <= ticker_tf.available_end_date):
            return pl.DataFrame()

        TICKER_DIR = os.path.join(self.data_dir, ticker)
        df = pl.DataFrame()

        for date in ts_range:
            month, year = str(date.month).zfill(2), date.year
            slug = f"{ticker}-{year}-{str(month).zfill(2)}.parquet"
            df_tmp = pl.read_parquet(os.path.join(TICKER_DIR, slug))
            df = df.vstack(df_tmp)

        return df.filter(
            (pl.col("time") >= pump_event.pump_time - lookback_delta) & 
            (pl.col("time") <= pump_event.pump_time)
        )
    
    def create_crosssection(
        self, pump_event: PumpEvent
    ) -> pd.DataFrame:
        """Creates a crosssection within a period of time of the pump. Applies create_features method to each dataset and
        combines all transformed data into a single dataframe representing one crosssection.
        """

        df_crosssection = pd.DataFrame()

        tickers = list(self.available_timeframes.keys())

        for ticker in tickers:
            df_ticker: pl.DataFrame = self.read_data(
                ticker=ticker, pump_event=pump_event, lookback_delta=timedelta(days=30)
            )

            if df_ticker.is_empty():
                continue

            df_features: pd.DataFrame = self.create_features(df_ticker=df_ticker, pump_event=pump_event)

            df_features["is_pumped"] = ticker == pump_event.ticker
            df_features["ticker"] = ticker

            df_crosssection = pd.concat([df_crosssection, df_features])

        return df_crosssection

    def multiprocess_transform_data(self) -> pd.DataFrame:
        df_transformed = pd.DataFrame()

        with Pool(processes=10) as pool:
            results = []

            for pump_event in self.labeled_pumps:
                res: AsyncResult = pool.apply_async(
                    partial(self.create_crosssection, pump_event=pump_event)
                )
                results.append(res)

            for result in tqdm(results):
                df_transformed = pd.concat([df_transformed, result.get()])

        return df_transformed


    def transform_data(self) -> pd.DataFrame:
        
        df_transformed = pd.DataFrame()

        for pump_event in tqdm(self.labeled_pumps):
            df_crosssection: pd.DataFrame = self.create_crosssection(pump_event=pump_event)
            df_transformed = pd.concat([df_transformed, df_crosssection])

        return df_transformed
  
    # @abstractmethod
    def create_features(
        self, pump_event: PumpEvent, df_ticker: pl.DataFrame
    ) -> pl.DataFrame:
        raise NotImplemented


if __name__ == "__main__":
    loader = DataLoader(
        data_dir="data/trades", labeled_pumps_file="data/pumps/cleaned/pumps_verified.csv"
    )

    loader.transform_data()
