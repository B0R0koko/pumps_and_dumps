from typing import *
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from tqdm import tqdm

import polars as pl
import pandas as pd
import os
import json
import re
import logging


logging.basicConfig(
    filename="features/pipes/cfg/execution.log",
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
)


@dataclass
class PumpEvent:
    pump_id: int
    ticker: str
    pump_time: str

    def __post_init__(self):
        self.pump_time = datetime.strptime(self.pump_time, "%Y-%m-%d %H:%M:%S")

    def __str__(self):
        return f"Pump event: {self.ticker} - {str(self.pump_time)}"


@dataclass
class TickerTimeframe:
    ticker: str
    start_time: datetime
    end_time: datetime

    def __repr__(self):
        return f"Ticker {self.ticker} is available from {str(self.start_time)} to {str(self.end_time)}"


class TaskStack:

    ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent

    def __init__(self, pump_events: List[PumpEvent]):
        self.pump_events = pump_events

    @property
    def pumps_config(self) -> List[Dict[str, str]]:
        data = []
        for pump_event in self.pump_events:
            data.append(
                {"ticker": pump_event.ticker, "pump_time": str(pump_event.pump_time)}
            )
        return data

    def pop(self, pump_event: PumpEvent) -> None:
        """Handles updates of the cfg/pumps.json file"""
        self.pump_events.remove(pump_event)
        # save to config file

        with open(
            os.path.join(self.ROOT_DIR, "features/pipes/cfg/pumps.json"), "w"
        ) as file:
            json.dump(self.pumps_config, file)


class DataLoader:

    ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent

    def __init__(self, data_dir: str, output_path: str) -> Self:
        self.data_dir: str = os.path.join(self.ROOT_DIR, data_dir)
        self.output_path: str = os.path.join(self.ROOT_DIR, output_path)

        self.available_timeframes: Dict[str, TickerTimeframe] = (
            self.get_tickers_available_timeframes()
        )

        self.pump_events: List[PumpEvent] = self.warmup_start()
        self.task_stack: TaskStack = TaskStack(pump_events=self.pump_events)

        self.tickers: List[str] = list(self.available_timeframes.keys())

    def warmup_start(self) -> List[PumpEvent]:
        """Load tasks to be completed"""
        with open(
            os.path.join(self.ROOT_DIR, "features/pipes/cfg/pumps.json"), "r"
        ) as file:
            pumps = json.load(file)
            return [
                PumpEvent(pump_id=i, ticker=pump["ticker"], pump_time=pump["pump_time"])
                for i, pump in enumerate(pumps)
            ]

    def create_pump_events(self, pumps: Dict[str, Dict[str, str]]) -> List[PumpEvent]:
        pump_events = [
            PumpEvent(ticker=pump["ticker"], pump_time=pump["pump_time"])
            for pump in pumps.values()
        ]
        return pump_events

    def get_tickers_available_timeframes(self) -> Dict[str, TickerTimeframe]:
        """Returns a list of all tickers collected and stored in self.data_dir directory.
        It defines start_month and end_month of data available for each ticker. This is needed when taking a crosssection
        of the pump."""

        available_timeframes = {}

        for ticker in os.listdir(self.data_dir):
            # folder is the name of the ticker
            TICKER_DIR = os.path.join(self.data_dir, ticker)
            parquet_files: List[str] = os.listdir(TICKER_DIR)

            available_dates: List[datetime] = [
                datetime.strptime(
                    re.search(r"-(\d{4}-\d{2}).parquet", file)[1], "%Y-%m"
                )
                for file in parquet_files
            ]

            start_date, end_date = min(available_dates), max(available_dates)

            available_timeframes[ticker] = TickerTimeframe(
                ticker=ticker, start_time=start_date, end_time=end_date
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

    def scan_data(
        self, ticker: str, pump_event: PumpEvent, lookback_delta: timedelta
    ) -> pl.LazyFrame:
        """Scans data from parquet files and performs initial transformation"""
        rb = pump_event.pump_time
        lb = rb - lookback_delta

        ts_range: List[datetime] = self.create_date_range(start=lb, end=rb)
        ticker_tf: TickerTimeframe = self.available_timeframes[ticker]

        if not (ticker_tf.start_time <= lb <= rb <= ticker_tf.end_time):
            return pl.LazyFrame()

        TICKER_DIR = os.path.join(self.data_dir, ticker)
        dfs = []

        for date in ts_range:
            month, year = str(date.month).zfill(2), date.year
            slug = f"{ticker}-{year}-{str(month).zfill(2)}.parquet"

            dfs.append(pl.scan_parquet(os.path.join(TICKER_DIR, slug)))

        return pl.concat(dfs, how="vertical").filter(
            (pl.col("time") >= pump_event.pump_time - lookback_delta)
            & (pl.col("time") <= pump_event.pump_time)
        )

    @abstractmethod
    def create_features(self, df_ticker: pl.DataFrame, pump_event: PumpEvent):
        return

    def create_crosssection(self, pump_event: PumpEvent) -> pd.DataFrame:
        """Creates a crosssection within a period of time of the pump. Applies create_features method to each dataset and
        combines all transformed data into a single dataframe representing one crosssection.
        """

        df_crosssection = pd.DataFrame()

        for ticker in tqdm(self.tickers, leave=False):
            df_ticker: pl.LazyFrame = self.scan_data(
                ticker=ticker, pump_event=pump_event, lookback_delta=timedelta(days=60)
            )
            df_ticker: pl.DataFrame = df_ticker.collect()

            if df_ticker.is_empty():
                # Skip iteration
                logging.debug(
                    f"Empty df_ticker. Pump {str(pump_event)}. While parsing ticker {ticker}"
                )
                continue

            df_features: pd.DataFrame = self.create_features(
                df_ticker=df_ticker, pump_event=pump_event
            )
            df_crosssection = pd.concat([df_crosssection, df_features])

        return df_crosssection

    def run(self) -> None:

        for pump_event in tqdm(self.pump_events):
            df_crosssection: pd.DataFrame = self.create_crosssection(
                pump_event=pump_event
            )

            output_path = Path(os.path.join(self.ROOT_DIR, self.output_path))

            if output_path.exists():
                df_crosssection.to_parquet(
                    output_path,
                    engine="fastparquet",
                    append=True,
                )
            else:
                df_crosssection.to_parquet(output_path, engine="fastparquet")

            # Update task stack
            self.task_stack.pop(pump_event=pump_event)


if __name__ == "__main__":
    loader = DataLoader(
        data_dir="data/trades", output_path="features/data/train.parquet"
    )
    loader.run()
