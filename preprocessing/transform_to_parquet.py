from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from functools import partial
from polars.io.csv.batched_reader import BatchedCsvReader
from typing import *


import polars as pl
import pandas as pd
import zipfile
import os
import re
import json
import gc


# Zipped data is stored without column names, which vary across exchanges, therefore
# we need this map to dynamically assign columns to dataframe for a specified exchange

COLS_EXCHANGES = {
    "binance": [
        "trade_id", "price", "qty",
        "quoteQty", "time", "isBuyerMaker",
        "isBestMatch",
    ],
    "kucoin": ["trade_id", "time", "price", "qty", "side"],
}

LEAVE_COLS = {
    "binance": ["price", "qty", "time", "isBuyerMaker"],
    "kucoin": ["time", "price", "qty", "side"],
}


class TaskStack:

    def __init__(self, tickers_to_preprocess: List[str], progress_file: str):
        # Kids, always remember to copy, I ran into some nasty bags with this
        self.tickers_to_preprocess: List[str] = tickers_to_preprocess.copy()
        self.progress_file: str = progress_file

    def pop(self, ticker: str) -> None:
        """Handles updates of the progress.json file"""
        self.tickers_to_preprocess.remove(ticker)
        # save to config file

        with open(os.path.join(self.progress_file), "w") as file:
            data = {"tickers_to_preprocess": self.tickers_to_preprocess}
            json.dump(data, file)


class ZipToParquetTransfromer:
    """
    Scrapy parser downloads data and stores it in zip files which is hard to work with, it is much
    easier to work with parquet files, therefore we need this transformer to handle conversion
    """

    def __init__(
        self,
        trades_dir: str = "data/trades",
        exchange: str = "binance",
        # output_dir will be used as output directory for all parquet files
        output_dir: str | None = "data/trades_parquet",
        transform_tickers: List[str] | None = None,
        require_split_into_days: bool = True,
        progress_file: str = "preprocessing/progress.json",
        warm_start: bool = False,
        n_workers: int = 1,
    ):
        self.exchange: str = exchange
        self.trades_dir: str = trades_dir
        self.output_dir: str = output_dir
        # for Kucoin is already split into daily zipped files, therefore no need to split it
        self.require_split_into_days: bool = require_split_into_days
        self.progress_file: str = progress_file
        self.n_workers: int = n_workers

        # Single process mode
        if self.n_workers == 1:

            if warm_start:
                # if warm_start is set to True, then use tickers that are left in progress.json
                self.tickers: List[str] = self.load_progress()
            else:
                # if transform_tickers is set to some list, use it, otherwise load all tickers from the folder
                self.tickers: List[str] = (
                    transform_tickers if transform_tickers else self.load_tickers()
                )
                # populate progress with the tickers that will be preprocessed
                self.populate_process(tickers=self.tickers)

            # Create TaskStack to keep track of the process and update progress.json by removing done jobs
            # (tickers)
            self.task_stack: TaskStack = TaskStack(
                tickers_to_preprocess=self.tickers, progress_file=self.progress_file
            )
        # Multiprocess mode
        else:
            assert (
                warm_start == False
            ), "Can't run warm_start=True in multiprocessing mode, reset n_workers to 1"
            self.tickers: List[str] = (
                transform_tickers if transform_tickers else self.load_tickers()
            )

    def populate_process(self, tickers: List[str]) -> None:
        with open(self.progress_file, "w") as file:
            data = {"tickers_to_preprocess": tickers}
            json.dump(data, file)

    def load_tickers(self) -> List[str]:
        """Get all tickers collect for a given exchange."""
        tickers: List[str] = os.listdir(os.path.join(self.trades_dir, self.exchange))
        return tickers

    def load_progress(self) -> List[str]:
        with open(self.progress_file, "r") as file:
            data = json.load(file)
        return data["tickers_to_preprocess"]

    def unzip_ticker(self, ticker: str):
        # /data/trades/binance/ADABTC
        ticker_dir: str = os.path.join(self.trades_dir, self.exchange, ticker)
        slugs: List[str] = [
            slug for slug in os.listdir(ticker_dir) if slug.endswith(".zip")
        ]

        # slugs are of the format ADABTC-trades-2023-01.zip
        pbar = tqdm(slugs, leave=False) if self.n_workers == 1 else slugs

        for slug in pbar:
            self.unzip_zipped_csv(ticker=ticker, slug=slug)

    def write_to_parquet(
        self, df: pl.DataFrame, output_ticker_dir: str, file_no_ext: str, ticker: str
    ) -> None:
        """Write data to parquet file"""
        # Group by data by days and write to parquet files
        df: pl.DataFrame = df.select(LEAVE_COLS[self.exchange])

        if not self.require_split_into_days:
            # just dump data into parquet file, data is already split into daily chunks
            df.write_parquet(
                os.path.join(output_ticker_dir, f"{file_no_ext}.parquet"),
                compression="gzip",
            )
            return

        # if we require a split perform a split and save to parquet files
        df = df.with_columns(
            time=pl.col("time").cast(pl.Datetime(time_unit="ms")).set_sorted()
        )
        df = df.with_columns(date=pl.col("time").dt.date())

        for date, df_date in df.group_by(["date"]):
            date: str = date[0].strftime("%Y-%m-%d")  # 2021-01-01
            file_path: str = os.path.join(
                # data/trades_parquet/ADABTC/ADABTC-trades-2021-01-01.parquet
                os.path.join(output_ticker_dir, f"{ticker}-trades-{date}.parquet")
            )
            # Drop date column and wrtie to parquet file
            df_date = df_date.drop(["date"])

            df_date.to_pandas().to_parquet(
                file_path, compression="gzip", engine="fastparquet", 
                append=os.path.exists(file_path), index=False
            )

        del df
        gc.collect()

    def unzip_zipped_csv(self, ticker: str, slug: str) -> None:

        # data/trades/binance/ADABTC/ADABTC-trades-2023-01.zip
        zip_file_path: str = os.path.join(self.trades_dir, self.exchange, ticker, slug)
        file_no_ext: str = re.search("(.*?).zip", slug)[1]

        # unzip data stored in the zipped archive
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            file_csv: str = zip_ref.namelist()[0]
            # Unzip csv file and load it to polars.DataFrame
            zip_ref.extract(
                file_csv,
                path="tmp",
            )

        file_path: str = os.path.join("tmp", f"{file_no_ext}.csv")

        output_ticker_dir: str = os.path.join(
            self.output_dir, self.exchange, ticker
        )
        # create output_dir if it doesn't exist and save parquet file to it
        os.makedirs(output_ticker_dir, exist_ok=True)

        if os.path.getsize(file_path) / 1024**2 <= 512:
            # If unpacked csv file is less than 512mb than load it in one go
            df: pl.DataFrame = pl.read_csv(
                source=file_path,
                has_header=False,
                new_columns=COLS_EXCHANGES[self.exchange],
            )
            self.write_to_parquet(
                df=df, output_ticker_dir=output_ticker_dir, ticker=ticker, file_no_ext=file_no_ext,
            )

        # Otherwise read file in chunks
        else:
            reader: BatchedCsvReader = pl.read_csv_batched(
                source=file_path,
                has_header=False,
                new_columns=COLS_EXCHANGES[self.exchange],
                low_memory=True,
                batch_size=100000
            )
            batch: List[pl.DataFrame] | None
            # Stop when chunk is None
            while (batch := reader.next_batches(1)) is not None:
                self.write_to_parquet(
                    df=batch[0],
                    output_ticker_dir=output_ticker_dir,
                    ticker=ticker,
                    file_no_ext=file_no_ext,
                )

        # remove unzipped csv file as it is loaded into the
        os.remove(os.path.join("tmp", file_csv))

    def transform_all(self):
        pbar = tqdm(self.tickers, total=len(self.tickers))

        for ticker in pbar:
            pbar.set_description(f"Transforming to parquet: {ticker}")
            self.unzip_ticker(ticker=ticker)

            # upon successful run, update progress.json file by removing from the array ticker
            # that has been preprocessed
            self.task_stack.pop(ticker=ticker)

    def transform_all_multiprocess(self):
        with (
            tqdm(
                total=len(self.tickers),
                desc="Transforming to parquet in multiprocessing mode: ",
            ) as pbar,
            Pool(processes=self.n_workers) as pool,
        ):
            results = []

            for ticker in self.tickers:
                res: AsyncResult = pool.apply_async(
                    partial(self.unzip_ticker, ticker=ticker)
                )
                results.append(res)

            for res in results:
                # Once result is obtained increment pbar
                res.get()
                pbar.update(1)

    def run(self) -> None:
        """
        Start trasnfromation to parquet files either with a single process or multiple depending on
        the value of n_workers
        """
        if self.n_workers > 1:
            self.transform_all_multiprocess()
        else:
            self.transform_all()


if __name__ == "__main__":
    transformer = ZipToParquetTransfromer(
        trades_dir="data/trades",
        exchange="binance",
        output_dir="data/trades_parquet",
        transform_tickers=["BTCUSDT"],  # set this to None, to have all tickers preprocessed from data/trades folder
        require_split_into_days=True,
        progress_file="preprocessing/progress.json",
        warm_start=False,  # set to True to run with warm_start, tickers will be loaded from progress.json
        n_workers=1,  # recommended to run in a single process, avoids memory overflows, enables state updates
    )
    transformer.run()
