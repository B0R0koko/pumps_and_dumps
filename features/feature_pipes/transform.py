from functools import partial
from multiprocessing.pool import AsyncResult
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing import Pool, freeze_support, RLock, current_process
from datetime import datetime, timedelta
from typing import *

import pandas as pd
import polars as pl
import os
import re


@dataclass
class PumpEvent:
    ticker: str
    pump_time: str

    def __post_init__(self):
        self.pump_time = datetime.strptime(self.pump_time, "%Y-%m-%d %H:%M:%S")


@dataclass
class DataTickerTimeFrame:
    ticker: str
    available_start_date: str
    available_end_date: str

    def __post_init__(self):
        self.available_start_date = pd.Timestamp(self.available_start_date)
        self.available_end_date = pd.Timestamp(self.available_end_date)


class DataLoader(ABC):
    """
    Parent class to handle feature creation. Inherit from this class and define abstract create_features method
    """

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

    def __init__(self, data_dir: str, labeled_pumps_file: str) -> None:
        self.data_dir = data_dir
        self.labeled_pumps_file = labeled_pumps_file

        self.labeled_pumps: List[PumpEvent] = self.load_pumps_csv()
        self.available_timeframes: List[DataTickerTimeFrame] = (
            self.map_existing_trade_data_to_time_intervals()
        )

    def load_pumps_csv(self) -> List[PumpEvent]:
        """
        Loads labeled pumps from data folder. Data contains fields: (ticker: str, time: str, source: str)
        """
        df = pd.read_csv(os.path.join(self.ROOT_DIR, self.labeled_pumps_file))
        pump_events: List[PumpEvent] = [
            PumpEvent(ticker=row.ticker, pump_time=row.time) for _, row in df.iterrows()
        ]
        return pump_events

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

            available_dates: List[pd.Timestamp] = [
                pd.Timestamp(re.search(r"-(\d{4}-\d{2}).parquet", file)[1])
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
    def create_date_range(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
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
                date_range.append(pd.Timestamp(year=year, month=month, day=1))

        return date_range

    def read_data(
        self, ticker: str, pump_event: PumpEvent, lookback_delta: timedelta
    ) -> pl.DataFrame:
        """
        Collect data for a ticker from the universe in a timeframe defined by pump_event.pump_time and lookback_delta.
        Returns a dataframe containing all trades for a given crypto in the timeline of the pump
        """
        # check if there is enough data collected to cover the whole timeperiod [pump-lookback: pump
        rb = pump_event.pump_time
        lb = rb - lookback_delta
        data_timeframe: DataTickerTimeFrame = self.available_timeframes[ticker]

        if not (
            data_timeframe.available_start_date
            <= lb
            <= rb
            <= data_timeframe.available_end_date
        ):
            return pl.DataFrame()

        rb = pump_event.pump_time
        lb = rb - lookback_delta

        ts_range = self.create_date_range(start=lb, end=rb)

        TICKER_DIR = os.path.join(self.ROOT_DIR, "data/trades", pump_event.ticker)
        df = pl.DataFrame()

        for date in ts_range:
            month, year = str(date.month).zfill(2), date.year
            slug = f"{pump_event.ticker}-{year}-{str(month).zfill(2)}.parquet"
            df_tmp = pl.read_parquet(os.path.join(TICKER_DIR, slug))

            df = df.vstack(df_tmp)

        return df

    def create_crosssection(self, pump_event: PumpEvent) -> pl.DataFrame:
        """Creates a crosssection within a period of time of the pump. Applies create_features method to each dataset and
        combines all transformed data into a single dataframe representing one crosssection.
        """

        df_crosssection = pd.DataFrame()
        tickers = list(self.available_timeframes.keys())[:3]

        pid = current_process().pid

        with tqdm(total=len(tickers), position=pid + 1) as pbar:

            for ticker in tickers:

                df_ticker: pl.DataFrame = self.read_data(
                    ticker=ticker,
                    pump_event=pump_event,
                    lookback_delta=timedelta(days=30),
                )

                if df_ticker.is_empty():
                    continue

                df_features: pd.DataFrame = self.create_features(
                    pump_event=pump_event, df_ticker=df_ticker
                )
                # add additional column labelling if these features represent the pumped ticker
                df_features["is_pumped"] = ticker == pump_event.ticker
                df_crosssection = pd.concat([df_crosssection, df_features])
                pbar.update(1)

        return df_crosssection

    # @abstractmethod
    def create_features(
        self, pump_event: PumpEvent, df_ticker: pd.DataFrame
    ) -> pl.DataFrame:
        raise NotImplemented

    def multiprocess_multiple(self) -> pd.DataFrame:
        freeze_support()

        df_transformed = pd.DataFrame()
        pool = Pool(processes=3, initargs=(RLock(),), initializer=tqdm.set_lock)
        results = []

        for i, pump_event in enumerate(self.labeled_pumps[:3]):
            res: AsyncResult = pool.apply_async(
                func=partial(self.create_crosssection, pump_event=pump_event)
            )
            results.append(res)

        pool.close()

        for result in results:
            df_transformed = pd.concat([df_transformed, result.get()])

        return df_transformed

    def multiprocess_transform(self) -> pd.DataFrame:

        df_transformed = pd.DataFrame()
        pool = Pool(processes=3)
        results = []

        for pump_event in self.labeled_pumps[:3]:
            res: AsyncResult = pool.apply_async(
                func=partial(self.create_crosssection, pump_event=pump_event)
            )
            results.append(res)

        pool.close()

        for result in results:
            df_transformed = pd.concat([df_transformed, result.get()])

        return df_transformed

    def multiprocess_transform_data(self) -> pd.DataFrame:
        df_transformed = pd.DataFrame()

        with (
            tqdm(
                total=len(self.labeled_pumps), desc="Transforming data", leave=False
            ) as pbar,
            Pool(processes=3) as pool,
        ):
            results = []

            for pump_event in self.labeled_pumps:
                res: AsyncResult = pool.apply_async(
                    partial(self.create_crosssection, pump_event=pump_event)
                )
                results.append(res)

            for result in results:
                df_transformed = pd.concat([df_transformed, result.get()])
                pbar.update(1)

        return df_transformed

    def transform_data(self) -> pd.DataFrame:
        df_transformed = pd.DataFrame()

        for pump_event in tqdm(self.labeled_pumps[:3]):

            df_crosssection: pd.DataFrame = self.create_crosssection(
                pump_event=pump_event
            )

            df_transformed = pd.concat([df_transformed, df_crosssection])

        return df_transformed


if __name__ == "__main__":
    print(
        DataLoader(
            "data/trades", "data/pumps/cleaned/pumps_verified.csv"
        ).available_timeframes
    )
