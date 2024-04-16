from urllib.parse import urlencode
from scrapy.http import Request
from datetime import datetime
from typing import *

import scrapy
import json
import pandas as pd
import os


CMC_ENDPOINT = (
    "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
)


START_DATE: str = "2017-01-01"
END_DATE: str = str(datetime.today().date())

OUTPUT_FILE_PATH: str = "data/cmc/cmc_snapshots.parquet"


class CMCParser(scrapy.Spider):
    name = "cmc_parser"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        for date in self.gen_time_range(START_DATE, END_DATE):
            params = {"convertId": "2781,1", "date": date, "limit": 1000, "start": 1}

            yield Request(
                url=CMC_ENDPOINT + "?" + urlencode(params),
                callback=self.parse_snapshot,
                meta={"snapshot": date},
            )

    def write_to_parquet_file(self, data: List[Dict[str, Any]]) -> None:
        """Write data to parquet file"""
        df_snapshot: pd.DataFrame = pd.DataFrame(data)

        df_snapshot.to_parquet(
            OUTPUT_FILE_PATH,
            engine="fastparquet",
            compression="gzip",
            append=os.path.exists(OUTPUT_FILE_PATH),
        )

    def parse_snapshot(self, response) -> None:
        data = json.loads(response.body)

        if "data" not in data:
            return

        parsed_data: List[Dict[str, Any]] = [
            {
                "name": crypto["name"],
                "symbol": crypto["symbol"],
                "slug": crypto["slug"],
                "cmc_rank": crypto["cmcRank"],
                "mcap_usdt": crypto["quotes"][0]["marketCap"],
                "mcap_btc": crypto["quotes"][1]["marketCap"],
                "snapshot": response.meta["snapshot"],
                "trading_volume_usdt": crypto["quotes"][0]["volume24h"],
                "trading_volume_btc": crypto["quotes"][1]["volume24h"],
            }
            for crypto in data["data"]
        ]

        self.write_to_parquet_file(data=parsed_data)
