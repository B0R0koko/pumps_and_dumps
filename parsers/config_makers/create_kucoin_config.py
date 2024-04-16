from pathlib import Path
from typing import *


import requests
import os
import re
import json


KUCOIN_URLS = [
    "https://historical-data.kucoin.com/?delimiter=/&prefix=data%2Fspot%2Fdaily%2Ftrades%2F",
    "https://historical-data.kucoin.com/?delimiter=/&prefix=data%2Fspot%2Fdaily%2Ftrades%2F&marker=data%2Fspot%2Fdaily%2Ftrades%2FOFNUSDT%2F",
]

CONFIG_PATH = "configs/kucoin.json"


def collect_tickers() -> List[str]:
    pattern = r"<Prefix>(.*?)</Prefix>"

    all_tickers = []

    # Collect all tickers from binance data vision
    for url in KUCOIN_URLS:
        resp = requests.get(url)
        tickers = re.findall(pattern=pattern, string=resp.text)
        tickers = [ticker.split("/")[-2] for ticker in tickers]

        all_tickers.extend(tickers)

    return all_tickers[1:]


def filter_tickers(tickers: List[str]) -> List[str]:
    # remove all tickers with DOWN and UP
    tickers = [
        ticker for ticker in tickers if all([el not in ticker for el in ["DOWN", "UP"]])
    ]
    # only against USDT, BUSD and BTC quotes
    tickers = [
        ticker
        for ticker in tickers
        if any([ticker.endswith(quote) for quote in ["BTC", "USDT", "BUSD"]])
    ]
    return tickers


def main() -> int:
    tickers: List[str] = collect_tickers()
    tickers: List[str] = filter_tickers(tickers=tickers)
    data = {"tickers_to_collect": tickers}

    with open(CONFIG_PATH, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
