from urllib.parse import urlencode
from scrapy.http import Request
from scrapy.exceptions import IgnoreRequest

from scrapy.utils.project import get_project_settings
from typing import *

import scrapy
import json
import pandas as pd


CMC_ENDPOINT = (
    "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
)


class CMCParser(scrapy.Spider):
    name = "cmc_parser"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.settings = get_project_settings()

    @staticmethod
    def gen_time_range(start_date: str, end_date: str) -> List[str]:
        """Generates timestamps with hourly delta between two timestamps"""
        ts_range = pd.date_range(
            start=pd.Timestamp(start_date),
            end=pd.Timestamp(end_date),
            freq="D",
        ).tolist()

        return [str(date.date()).replace("-", "") for date in ts_range]

    def start_requests(self) -> Iterable[Request]:
        """Collect data on historical market caps from CMC"""
        start_date: str = self.settings.get("START_DATE")
        end_date: str = self.settings.get("END_DATE")

        for date in self.gen_time_range(start_date, end_date):
            params = {"convertId": "2781,1", "date": date, "limit": 1000, "start": 1}

            yield Request(
                url=CMC_ENDPOINT + "?" + urlencode(params),
                callback=self.parse_snapshot,
                meta={"snapshot": date},
            )

    def parse_snapshot(self, response) -> Dict[str, Any]:
        data = json.loads(response.body)

        if "data" not in data:
            return

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
