from scrapy.exceptions import IgnoreRequest
from scrapy.utils.project import get_project_settings
from urllib.parse import urlencode
from typing import *

import scrapy
import json


def generate_year_months(dt_from: str, dt_to: str) -> List[Tuple[str, str]]:
    """Generate lists of month-year tuples"""

    year_from, month_from = map(int, dt_from.split("-"))
    year_to, month_to = map(int, dt_to.split("-"))

    ym_tuples: List[Tuple[int, int]] = []
    cur_year, cur_month = year_from, month_from

    while True:
        ym_tuples.append(
            (
                cur_month if cur_month >= 10 else f"0{cur_month}",  # format 1 to 01
                cur_year,
            )
        )
        cur_month += 1

        if cur_month > 12:
            cur_month = 1
            cur_year += 1

        if cur_year > year_to or (cur_year == year_to and cur_month > month_to):
            break

    return ym_tuples


class BinanceTradesSpider(scrapy.Spider):
    name = "trades_spider"

    custom_settings = {
        "ITEM_PIPELINES": {
            "pumps.pipelines.parquet_pipeline.ParquetPipeline": 1,
        }
    }

    def __init__(self, **kwargs) -> Self:
        super().__init__(**kwargs)
        self.settings = get_project_settings()

        self.tickers: List[Dict[str, Any]] = self.load_config()

    def load_config(self) -> List[Dict[str, Any]]:
        with open(self.settings.get("CONFIG_PATH"), "r") as file:
            tickers = json.load(file)
            return tickers

    def start_requests(self) -> scrapy.Request:
        for ticker_cfg in self.tickers:
            ticker = ticker_cfg["symbol"]
            # Start with checking if this trading pair existed on Binance
            url = (
                self.settings.get("BINANCE_ENDPOINT")
                + "?"
                + urlencode({"symbol": f"{ticker}", "fromId": 0})
            )

            yield scrapy.Request(
                url=url, callback=self.query_data, meta={"config": ticker_cfg}
            )

    def query_data(self, response) -> scrapy.Request:
        ticker, dt_from, dt_to = response.meta["config"].values()

        # If either symbol doesn't exist on Binance or data already has been downloaded, them cancel downloading
        if response.status != 200:
            raise IgnoreRequest()

        time_intervals: List[Tuple[int, int]] = generate_year_months(
            dt_from=dt_from, dt_to=dt_to
        )

        for month, year in time_intervals:
            endpoint = self.settings.get("SOURCE_URL").format(
                ticker, ticker, year, month
            )

            yield scrapy.Request(
                url=endpoint,
                callback=self.write_data,
                meta={"ticker": ticker, "slug": f"{ticker}-{year}-{month}"},
            )

    def write_data(self, response) -> Dict[str, Any]:
        """if you want to write to zip files change to another pipeline in settings.py"""
        ticker, slug = response.meta["ticker"], response.meta["slug"]

        yield {"ticker": ticker, "slug": slug, "data": response.body}
