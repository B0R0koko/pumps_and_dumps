from pathlib import Path
from typing import *
from scipy.stats import chi2
from datetime import datetime, timedelta
from feature_pipes.transform import DataLoader, PumpEvent

import numpy as np
import pandas as pd
import polars as pl
import os
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader(DataLoader):

    def transform_to_features(
        self, df_group: pd.DataFrame, pump_event: PumpEvent, window: str
    ) -> pd.DataFrame:

        # Volume imbalance
        df_group = df_group.with_columns(
            is_whale=pl.col("qty_abs")
            >= pl.col("qty_abs").rolling_quantile(
                quantile=0.999, by="time", window_size=window
            ),
            is_buy=pl.col("qty_sign") > 0,
        )

        df_group = df_group.with_columns(
            whale_imbalance_ratio=(
                (pl.col("qty_sign") * pl.col("is_whale")).rolling_sum(
                    by="time", window_size=window
                )
                / (pl.col("qty_abs") * pl.col("is_whale")).rolling_sum(
                    by="time", window_size=window
                )
            ),
            regular_imbalance_ratio=(
                (pl.col("qty_sign") * (1 - pl.col("is_whale"))).rolling_sum(
                    by="time", window_size=window
                )
                / (pl.col("qty_abs") * (1 - pl.col("is_whale"))).rolling_sum(
                    by="time", window_size=window
                )
            ),
        )

        # Datetime features
        df_group = df_group.with_columns(weekday=pl.col("time").dt.weekday())

        # prepend dummies dataframe to the right
        weekday_dummmies = df_group["weekday"].to_dummies()
        weekday_cols = weekday_dummmies.columns

        df_group = df_group.with_columns_seq(weekday_dummmies)

        df_group = df_group.with_columns(
            whale_vol=(pl.col("is_whale") * pl.col("qty_abs")).rolling_sum(
                by="time", window_size=window
            ),
            whale_num_trade=pl.col("is_whale")
            .cast(pl.Float32)
            .rolling_sum(by="time", window_size=window),
        )

        df_group = df_group.with_columns_seq(
            df_group.rolling(index_column="time", period=window).agg(
                [
                    # weekday_dummy * is_whale * qty_abs
                    (pl.col(col) * pl.col("is_whale") * pl.col("qty_abs")).sum()
                    for col in weekday_cols
                ]
            )
        )

        df_group = df_group.with_columns_seq(
            [pl.col(f"weekday_{i}") / pl.col("whale_vol") for i in range(1, 8)]
        )

        # Hour bins

        df_group = df_group.with_columns(hour=pl.col("time").dt.hour())
        # Bins hours
        bin_hours = [3, 6, 9, 12, 15, 18, 21]
        labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]
        # Create bins for hours
        df_group = df_group.with_columns(
            hour_bin=pl.col("hour").cut(
                breaks=bin_hours, labels=labels, left_closed=True
            )
        )

        # Create dummies for hour bin of the trades
        hour_bins_dummies = df_group["hour_bin"].to_dummies()
        hour_bins_cols = hour_bins_dummies.columns

        df_group = df_group.with_columns_seq(hour_bins_dummies)

        # calculate hour bin whale volume
        df_group = df_group.with_columns_seq(
            df_group.rolling(index_column="time", period=window).agg(
                [
                    # hour_bin_dummy * is_whale * qty_abs
                    (pl.col(col) * pl.col("is_whale") * pl.col("qty_abs")).sum()
                    for col in hour_bins_cols
                ]
            )
        )

        # divide by the overall traded volume
        df_group = df_group.with_columns_seq(
            [pl.col(col) / pl.col("whale_vol") for col in hour_bins_cols]
        )

        # Benford's law features
        df_group = df_group.with_columns(
            qty_first_digit=pl.col("qty_abs").cast(str).str.slice(0, 1).cast(pl.Int8)
        )
        # Similarily create dummies for each first digit
        df_group = df_group.filter(pl.col("qty_first_digit") > 0)

        first_digits_dummies = df_group["qty_first_digit"].to_dummies()
        first_digit_cols = first_digits_dummies.columns

        df_group = df_group.with_columns_seq(first_digits_dummies)

        df_group = df_group.with_columns_seq(
            df_group.rolling(index_column="time", period=window).agg(
                [
                    # hour_bin_dummy * is_whale * qty_abs
                    (pl.col(col) * pl.col("is_whale")).sum()
                    for col in first_digit_cols
                ]
            )
        )

        # logarithmic distribution Benford's law
        expected_freq = pl.Series([np.log(i + 1) - np.log(i) for i in range(1, 10)])

        # add expected frequencies
        df_group = df_group.with_columns(
            *[
                # expected frequency = distrib * whale_num_trades so far
                (expected_freq[i] * pl.col("whale_num_trade")).alias(
                    f"expected_freq_digit_{i+1}"
                )
                for i, col in enumerate(first_digit_cols)
            ]
        )

        expected_freq_cols = [f"expected_freq_digit_{i}" for i in range(1, 10)]

        # calculate chi_squared statistic
        df_group = df_group.with_columns(
            chi2_stat=(
                (df_group[first_digit_cols] - df_group[expected_freq_cols])
                / df_group[expected_freq_cols]
            )
            .sum_horizontal()
            .pow(2)
        )

        features = (
            ["time", "whale_imbalance_ratio", "regular_imbalance_ratio", "chi2_stat"]
            + hour_bins_cols
            + weekday_cols
        )

        df_features = df_group[features]

        df_features = df_features.rolling(index_column="time", period="1d").agg(
            whale_imbalance_ratio_mean=pl.col("whale_imbalance_ratio").mean(),
            regular_imbalance_ratio_mean=pl.col("regular_imbalance_ratio").mean(),
            # Weekday cols, median - max
            *[pl.col(col).median().alias(f"{col}_median") for col in weekday_cols],
            *[pl.col(col).max().alias(f"{col}_max") for col in weekday_cols],
            # Hours cols, median - max,
            *[pl.col(col).median().alias(f"{col}_median") for col in hour_bins_cols],
            *[pl.col(col).max().alias(f"{col}_max") for col in hour_bins_cols],
            # Pvalue
            *[
                pl.col("chi2_stat").median().alias("chi2_stat_median"),
                pl.col("chi2_stat").max().alias("chi2_stat_max"),
            ],
        )

        # select 24*14 hours before
        df_res = pl.DataFrame()

        for i in range(1, 24 * 14 + 1):
            after_threshold = df_features["time"] >= pump_event.pump_time - timedelta(
                minutes=10
            ) - timedelta(hours=i)

            df_res = pl.concat([df_res, df_features.filter(after_threshold)[0]])

        df_res = df_res.drop(columns=["time"])

        df_res = df_res.to_pandas()

        df_res.index = [f"{i}_H" for i in range(1, 24 * 14 + 1)]
        df_res = df_res.unstack().to_frame().sort_index(level=1).T
        df_res.columns = df_res.columns.map("_".join)

        return df_res

    def create_features(
        self, pump_event: PumpEvent, df_ticker: pl.DataFrame
    ) -> pd.DataFrame:
        # Perform all feature engineering here
        # Leave only the last 24 hours of the data before pump

        df_ticker = df_ticker.filter(
            (pl.col("time") < pump_event.pump_time)
            & (pl.col("time") >= pump_event.pump_time - timedelta(days=30))
        )
        df_ticker = df_ticker.with_columns(
            ((1 - 2 * pl.col("isBuyerMaker").cast(pl.Int8)) * pl.col("qty")).alias(
                "qty_sign"
            )
        )

        df_group = df_ticker.group_by("time", maintain_order=True).agg(
            isBuyerMaker=pl.col("isBuyerMaker").mean(),
            qty_abs=pl.col("qty").sum(),
            qty_sign=pl.col("qty_sign").sum(),
        )

        df_group = df_group.with_columns(time=pl.col("time").set_sorted())

        df_features: pd.DataFrame = self.transform_to_features(
            df_group=df_group, pump_event=pump_event, window="7d"
        )

        return df_features


if __name__ == "__main__":

    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

    loader = Loader(
        data_dir="data/trades",
        labeled_pumps_file="data/pumps/cleaned/pumps_verified.csv",
    )

    df_train: pd.DataFrame = loader.multiprocess_multiple()
    df_train.to_parquet(os.path.join(ROOT_DIR, "data/data_transformed.parquet"))
