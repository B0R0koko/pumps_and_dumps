import pandas as pd
import os
import json
import re
import numpy as np

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import *


ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent


class DataTransformer:
    def __init__(self, pumps_path: str, data_dir: str):
        self.pumps_path = pumps_path
        self.data_dir = data_dir

        self.ticker_available_timeframe = self.map_stored_data()

        self.load_pumps_config()

    def load_pumps_config(self):
        with open(os.path.join(ROOT_DIR, self.pumps_path), "r") as file:
            self.pumps = json.load(file)

    def map_stored_data(self) -> List[Dict[str, str]]:
        """Go through data collected in data folder and create time intervals to know precisely between each dates
        ticker has been traded on binance
        """
        data_dir = os.path.join(ROOT_DIR, self.data_dir)

        ticker_available_timeframe = []

        for ticker_dir in os.listdir(data_dir):
            ticker_dates: List[datetime] = []

            for file in os.listdir(os.path.join(data_dir, ticker_dir)):
                date = re.search("-(\d{4}-\d{2}).parquet", file)[1]
                ticker_dates.append(pd.Timestamp(date))

            min_date, max_date = min(ticker_dates), max(ticker_dates)

            ticker_available_timeframe.append(
                {"ticker": ticker_dir, "date_from": min_date, "date_to": max_date}
            )

        return ticker_available_timeframe

    def read_file(self, time_range: List[Tuple[int, int]], ticker: str) -> pd.DataFrame:
        df = pd.DataFrame()

        for time_period in time_range:
            month, year = time_period
            # 1INCHBTC/1INCHBTC-2021-09.parquet
            slug = f"{ticker}/{ticker}-{year}-{str(month).zfill(2)}.parquet"
            df_month = pd.read_parquet(os.path.join(ROOT_DIR, self.data_dir, slug))
            df = pd.concat([df, df_month], axis=0)

        return df

    def create_crosssection(
        self, pump_event: Dict[str, Any], lookback_months: float = 1
    ):
        ticker, time = pump_event["ticker"], pd.Timestamp(pump_event["time"])
        lb = time - pd.Timedelta(days=30 * lookback_months)

        ts_range = pd.date_range(
            start=lb, end=time, freq="MS", inclusive="both", normalize=True
        ).tolist()

        time_range = [(el.month, el.year) for el in ts_range]

        df_pumped: pd.DataFrame = self.read_file(time_range=time_range, ticker=ticker)

        # load the universe (other coins within this crosssection)
        for ticker in self.ticker_available_timeframe:
            if ticker["date_from"] <= ts_range[0] <= ts_range[-1] <= ticker["date_to"]:
                df_other_ticker: pd.DataFrame = self.read_file(
                    time_range=time_range, ticker=ticker
                )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """create data pipeline here"""
        df_buy = (
            df[df.side == "buy"].resample("5min", on="time")["qty"].sum().to_frame()
        )
        df_sell = (
            df[df.side == "sell"].resample("5min", on="time")["qty"].sum().to_frame()
        )
        df_vol = df_buy.merge(
            df_sell, on="time", how="outer", suffixes=["_buy", "_sell"]
        ).fillna(0)

        df_vol = df_vol.sort_values(by="time", ascending=True)
        df_vol.drop

        return df_vol

    def create_train_data(self):
        for pump_event in self.pumps:
            self.create_crosssection(pump_event=pump_event)
