from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from dotenv import load_dotenv

from ..models import Ticker, Trade, Base

from typing import *

import pytz
import zipfile
import io
import csv
import os


# Load database credentials and compose a postgres uri
load_dotenv()

DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME = [
    os.getenv(var)
    for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
]

DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def preprocess_row(trade: List[str], ticker_obj: Ticker) -> Trade:
    return Trade(
        trade_id=int(trade[0]),
        price=float(trade[1]),
        qty=float(trade[2]),
        time=datetime.fromtimestamp(int(trade[4]) / 1000, tz=pytz.utc),
        isBuyerMaker=trade[5] == "True",
        ticker_id=ticker_obj.id,
    )


BATCH_SIZE = 50000


class DBPipeline:

    """Write data to postgres database"""

    def open_spider(self, spider):
        engine: Engine = create_engine(DB_URI)
        session_maker: sessionmaker[Session] = sessionmaker(
            bind=engine, expire_on_commit=False
        )
        self.session: Session = session_maker()

        Base.metadata.create_all(engine)

    def find_ticker(self, ticker_name: str) -> Ticker | None:
        return self.session.query(Ticker).filter(Ticker.name == ticker_name).first()

    def process_item(self, response, spider):
        data, ticker_name = response["data"], response["ticker"]

        # Unzip data and use csv.reader
        with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref:
            for file_name in zip_ref.namelist():
                file_content = io.StringIO(zip_ref.read(file_name).decode("utf-8"))
                reader = csv.reader(file_content)

        # If this ticker doesn't exist in tickers table, then create it and add it to commit
        ticker_obj: Ticker | None = self.find_ticker(ticker_name=ticker_name)

        if not ticker_obj:
            ticker_obj = Ticker(name=ticker_name)
            self.session.add(ticker_obj)
            self.session.commit()

        trades: List[Trade] = []

        for i, trade in enumerate(reader):
            trades.append(preprocess_row(trade, ticker_obj=ticker_obj))

            if i % BATCH_SIZE == 0:
                self.session.bulk_save_objects(trades)
                self.session.commit()
                trades.clear()

    def close_spider(self, spider):
        self.session.close()
