from scrapy.utils.project import get_project_settings
from scrapy.exceptions import IgnoreRequest
from urllib.parse import urljoin, urlencode

from typing import *

import scrapy
import pandas as pd
import json


TRADES_ENDPOINT = (
    "https://data.binance.vision/data/spot/monthly/trades/{}/{}-trades-{}-{}.zip"
)
BINANCE_ENDPOINT = "https://api.binance.com/api/v3/historicalTrades"


class KlineSpider(scrapy.Spider):
    name = "kline_parser"

    custom_settings = {
        "ITEM_PIPELINES": {
            "parsers.pipes.parquet_pipe.ParquetPipeline": 1,
        }
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.settings = get_project_settings()
        self.tickers: List[str] = self.load_config()

    def load_config(self) -> List[str]:
        with open(self.settings.get("CONFIG_PATH"), "rb") as file:
            return json.load(file)["tickers"]

    @staticmethod
    def gen_time_range(start_date: str, end_date: str) -> List[pd.Timestamp]:
        """Generates timestamps with hourly delta between two timestamps"""
        ts_range = pd.date_range(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            freq="MS",
            inclusive="both",
            normalize=True,
        ).tolist()
        return ts_range

    def start_requests(self) -> scrapy.Request:
        for ticker in self.tickers:
            # Start with checking if this trading pair existed on Binance
            url = f'{BINANCE_ENDPOINT}?{urlencode({"symbol": ticker, "fromId": 0})}'
            yield scrapy.Request(
                url=url, callback=self.get_first_trades, meta={"ticker": ticker}
            )

    def get_first_trades(self, response) -> scrapy.Request:
        ticker = response.meta["ticker"]

        # If either symbol doesn't exist on Binance or data already has been downloaded, them cancel downloading
        if response.status != 200:
            raise IgnoreRequest()

        data = json.loads(response.body)
        ts_trade_start = pd.Timestamp(data[0]["time"], unit="ms")
        # do not provide fromId as a parameter, API will return the latest trades and we will be able to see the last
        # date ticker was traded
        url = f'{BINANCE_ENDPOINT}?{urlencode({"symbol": ticker})}'

        yield scrapy.Request(
            url=url,
            callback=self.query_data,
            meta={"ticker": ticker, "ts_trade_start": ts_trade_start},
        )

    def query_data(self, response) -> scrapy.Request:
        ticker, ts_trade_start = (
            response.meta["ticker"],
            response.meta["ts_trade_start"],
        )

        data = json.loads(response.body)
        ts_trade_last = pd.Timestamp(data[-1]["time"], unit="ms")

        ts_list: List[pd.Timestamp] = self.gen_time_range(
            start_date=ts_trade_start, end_date=ts_trade_last
        )

        for date in ts_list:

            year, month = date.year, str(date.month).zfill(2)

            yield scrapy.Request(
                url=TRADES_ENDPOINT.format(ticker, ticker, year, month),
                callback=self.write_data,
                meta={"ticker": ticker, "slug": f"{ticker}-{year}-{month}"},
            )

    def write_data(self, response) -> Dict[str, Any]:
        """if you want to write to zip files change to another pipeline in settings.py"""
        ticker, slug = response.meta["ticker"], response.meta["slug"]

        yield {"ticker": ticker, "slug": slug, "data": response.body}
