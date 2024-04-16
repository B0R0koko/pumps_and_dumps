from scrapy.utils.project import get_project_settings
from typing import *


import scrapy
import re
import json


BINANCE_API_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/spot/monthly/trades/"
BINANCE_ROOT = "https://data.binance.vision/"

CONFIG_PATH = "configs/binance.json"


class BinanceTradeParser(scrapy.Spider):
    """Collect data from Binance data vision api"""

    name = "binance_parser"

    custom_settings = {
        "ITEM_PIPELINES": {
            "parsers.trades.pipes.zip.ZipPipeline": 1,
        },
        "OUTPUT_DIR": "data/trades/binance",
    }

    def __init__(self) -> Self:
        super().__init__()
        self.settings = get_project_settings()
        self.tickers: List[str] = self.load_tickers()

    def load_tickers(self) -> List[str]:
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
        return config["tickers_to_collect"]

    def start_requests(self) -> Iterable[scrapy.Request]:
        """For each ticker query the page with all zip files laid out by months"""
        for ticker in self.tickers:
            yield scrapy.Request(
                url=f"{BINANCE_API_URL}{ticker}/",
                callback=self.parse_ticker,
                meta={
                    "ticker": ticker,
                },
            )

    def parse_ticker(self, response):
        """Extract all zip file handles associated with this ticker and collect the data"""
        hrefs: List[str] = re.findall(pattern=r"<Key>(.*?)</Key>", string=response.text)
        hrefs = [href for href in hrefs if "CHECKSUM" not in href]

        for href in hrefs:

            data_url = f"{BINANCE_ROOT}{href}"
            slug = re.search(r"(.*?).zip", href.split("/")[-1])[1]

            yield scrapy.Request(
                url=data_url,
                callback=self.write_data,
                meta={"ticker": response.meta["ticker"], "slug": slug},
            )

    def write_data(self, response):
        """Pass collected data to Pipeline"""
        yield {
            "response": response,
            "ticker": response.meta["ticker"],
            "slug": response.meta["slug"],
        }
