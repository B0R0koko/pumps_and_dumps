from urllib.parse import urlencode
from selectolax.parser import HTMLParser
from scrapy.http import Request

from typing import *

import scrapy
import json


class CMCParser(scrapy.Spider):
    name = "cmc_parser"

    def start_requests(self) -> Request:
        """Query CMC historical page and parse all of the hrefs heading to each snapshot"""
        yield Request(
            url="https://coinmarketcap.com/historical/",
            callback=self.parse_available_snapshots,
        )

    def parse_available_snapshots(self, response) -> Iterable[Request]:
        dom = HTMLParser(response.body)

        snapshots: List[str] = [
            el.attrs["href"].split("/")[-2]
            for el in dom.css(".cmc-main-section__content a.historical-link")
        ]

        for snapshot in snapshots:
            params = {
                "convertId": "2781,1",
                "date": snapshot,
                "limit": 1000,
                "start": 1,
            }

            yield Request(
                url="https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical" + "?" + urlencode(params),
                callback=self.parse_snapshot,
                meta={"snapshot": snapshot},
            )

    def parse_snapshot(self, response) -> Dict[str, Any]:
        data = json.loads(response.body)

        parsed_data = [
            {
                "name": crypto["name"],
                "symbol": crypto["symbol"],
                "slug": crypto["slug"],
                "cmcRank": crypto["cmcRank"],
                "mcap": crypto["quotes"][0]["marketCap"],
                "snapshot": response.meta["snapshot"],
            }
            for crypto in data["data"]
        ]

        for crypto in parsed_data:
            yield crypto
