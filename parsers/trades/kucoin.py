from typing import *

import scrapy
import re
import json


KUCOIN_API_URL = "https://historical-data.kucoin.com/?delimiter=/&prefix=data%2Fspot%2Fdaily%2Ftrades%2F{}%2F"
KUCOIN_ROOT = "https://historical-data.kucoin.com/"


CONFIG_PATH = "configs/kucoin.json"


class KucoinTradeParser(scrapy.Spider):
    """Collect data from Kucoin api"""

    name = "kucoin_parser"

    custom_settings = {
        "ITEM_PIPELINES": {
            "parsers.trades.pipes.zip.ZipPipeline": 1,
        },
        "OUTPUT_DIR": "data/trades/kucoin",
    }

    def __init__(self) -> Self:
        super().__init__()
        self.tickers: List[str] = self.load_tickers()

    def load_tickers(self) -> List[str]:
        with open(CONFIG_PATH, "r") as file:
            config = json.load(file)
        return config["tickers_to_collect"]

    def start_requests(self) -> Iterable[scrapy.Request]:
        """For each ticker query the page with all zip files laid out by months"""
        for ticker in self.tickers:
            yield scrapy.Request(
                url=KUCOIN_API_URL.format(ticker),
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

            data_url = f"{KUCOIN_ROOT}{href}"
            slug = re.search(r"(.*?).zip", href.split("/")[-1])[1]

            yield scrapy.Request(
                url=data_url,
                callback=self.write_data,
                meta={"ticker": response.meta["ticker"], "slug": slug},
            )

    def write_data(self, response):
        """Pass collected data to Pipeline. Zip pipeline is prefered as it ts the fastest one"""
        yield {
            "response": response,
            "ticker": response.meta["ticker"],
            "slug": response.meta["slug"],
        }
