from typing import *
from datetime import timedelta
from pipes.dataloader import PumpEvent

import polars as pl
import pandas as pd
import numpy as np


def transform_features(
    df: pl.DataFrame, pump_event: PumpEvent, window: str, window_output: str
) -> pd.DataFrame:
    df_group: pl.LazyFrame = (
        df.lazy()
        .with_columns(
            ((1 - 2 * pl.col("isBuyerMaker").cast(pl.Int8)) * pl.col("qty")).alias(
                "qty_sign"
            )
        )
        .group_by("time", maintain_order=True)
        .agg(
            qty_abs=pl.col("qty").sum(),
            qty_sign=pl.col("qty_sign").sum(),
        )
        .with_columns(time=pl.col("time").set_sorted())
        .with_columns(date=pl.col("time").dt.truncate("1d"))
    )

    df_quantiles: pl.LazyFrame = df_group.group_by_dynamic(
        index_column="time", every="1d"
    ).agg(qty_quantile=pl.col("qty_abs").quantile(0.999))

    df_group: pl.LazyFrame = (
        df_group.join(df_quantiles, left_on="date", right_on="time", how="left")
        .with_columns(
            is_whale=(pl.col("qty_abs") >= pl.col("qty_quantile")).cast(pl.Int8),
            is_buy=(pl.col("qty_sign") > 0).cast(pl.Int8),
        )
        .with_columns(
            # Whale volumes
            rolling_qty_abs_whale_vol=(
                pl.col("qty_abs") * pl.col("is_whale")
            ).rolling_sum(by="time", window_size=window),
            rolling_qty_sign_whale_vol=(
                pl.col("qty_sign") * pl.col("is_whale")
            ).rolling_sum(by="time", window_size=window),
            # Regular volumes
            rolling_qty_abs_reg_vol=(
                pl.col("qty_abs") * ~pl.col("is_whale")
            ).rolling_sum(by="time", window_size=window),
            rolling_qty_sign_reg_vol=(
                pl.col("qty_sign") * pl.col("is_whale")
            ).rolling_sum(by="time", window_size=window),
        )
        .with_columns(
            whale_imbalance_ratio=pl.col("rolling_qty_sign_whale_vol")
            / pl.col("rolling_qty_abs_whale_vol"),
            regular_imbalance_ratio=pl.col("rolling_qty_sign_reg_vol")
            / pl.col("rolling_qty_abs_reg_vol"),
        )
    )

    df_group: pl.DataFrame = df_group.collect()

    # ---------------------------------------------------------------------------------------------------------------
    # --Weekday features---------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    df_weekday: pl.DataFrame = (
        df_group.with_columns(weekday=pl.col("time").dt.weekday())
        .select("weekday")
        .to_dummies()
    )

    # check if all weekdays are present
    present_weekdays = set([int(col.split("_")[-1]) for col in df_weekday.columns])
    all_weekdays = set(range(1, 8))
    missing_weekdays = all_weekdays - present_weekdays

    df_weekday: pl.DataFrame = df_weekday.with_columns_seq(
        [pl.lit(0).alias(f"weekday_{weekday}") for weekday in missing_weekdays]
    )

    weekday_cols = df_weekday.columns

    df_weekday = df_weekday.with_columns(
        is_whale=df_group["is_whale"],
        qty_abs=df_group["qty_abs"],
        is_buy=df_group["is_buy"],
        time=df_group["time"],
    )

    df_weekday: pl.DataFrame = (
        df_weekday.lazy()
        # create running volume for buy whale trades
        .with_columns(
            rolling_whale_abs_buy_vol=(
                pl.col("is_whale") * pl.col("is_buy") * pl.col("qty_abs")
            ).rolling_sum(window_size=window, by="time")
        )
        # multiply dummy for each day buy abs vol
        .with_columns_seq(
            [
                (
                    pl.col(col)
                    * pl.col("is_whale")
                    * pl.col("is_buy")
                    * pl.col("qty_abs")
                ).rolling_sum(window_size=window, by="time")
                for col in weekday_cols
            ]
        )
        .with_columns_seq(
            [pl.col(col) / pl.col("rolling_whale_abs_buy_vol") for col in weekday_cols]
        )
        .select(weekday_cols)
    )

    # collect data from LazyFrame
    df_group = df_group.with_columns_seq(df_weekday.collect())

    # ---------------------------------------------------------------------------------------------------------------
    # --Hours features-----------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    bin_hours = [3, 6, 9, 12, 15, 18, 21]
    labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]

    df_hours: pl.DataFrame = (
        df_group.with_columns(hour=pl.col("time").dt.hour())
        .select("hour")
        .to_series()
        .cut(breaks=bin_hours, labels=labels, left_closed=True)
        .to_dummies()
        # add columns needed
        .with_columns(
            is_whale=df_group["is_whale"],
            qty_abs=df_group["qty_abs"],
            is_buy=df_group["is_buy"],
            time=df_group["time"],
        )
    )

    present_cols = set(df_hours.columns)
    hour_cols = set([f"hour_{label}" for label in labels])
    missing_cols = hour_cols - present_cols

    df_hours = (
        df_hours
        # fill in missing bins
        .with_columns_seq([pl.lit(0).alias(bin) for bin in missing_cols])
    )

    df_hours: pl.LazyFrame = (
        df_hours.lazy()
        # create running volume for buy whale trades
        .with_columns(
            rolling_whale_abs_buy_vol=(
                pl.col("is_whale") * pl.col("is_buy") * pl.col("qty_abs")
            ).rolling_sum(window_size=window, by="time")
        )
        # multiply dummy for each day buy abs vol
        .with_columns_seq(
            [
                (
                    pl.col(col)
                    * pl.col("is_whale")
                    * pl.col("is_buy")
                    * pl.col("qty_abs")
                ).rolling_sum(window_size=window, by="time")
                for col in hour_cols
            ]
        )
        .with_columns_seq(
            [pl.col(col) / pl.col("rolling_whale_abs_buy_vol") for col in hour_cols]
        )
        .select(hour_cols)
    )

    df_group: pl.DataFrame = df_group.with_columns_seq(df_hours.collect())

    # ---------------------------------------------------------------------------------------------------------------
    # --Benfard's Law -----------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    df_first_digits = (
        df_group.with_columns(
            first_digit=pl.col("qty_abs").cast(str).str.slice(0, 1).cast(pl.Int8)
        )
        .select("first_digit")
        .to_dummies()
    )

    # Drop 0 digit as we shouldn't observe quantities of 0
    df_first_digits = df_first_digits.drop("first_digit_0")

    # add missing digits
    present_digits = set(df_first_digits.columns)
    all_digits = set([f"first_digit_{i}" for i in range(1, 10)])

    missing_digits = all_digits - present_digits

    df_first_digits = df_first_digits.with_columns_seq(
        [pl.lit(0).alias(col) for col in missing_digits]
    )

    digit_cols = df_first_digits.columns

    df_first_digits = df_first_digits.with_columns(
        is_whale=df_group["is_whale"],
        qty_abs=df_group["qty_abs"],
        is_buy=df_group["is_buy"],
        time=df_group["time"],
    )

    df_first_digits = df_first_digits.with_columns(
        rolling_num_whale_trades=pl.col("is_whale")
        .cast(pl.Float32)
        .rolling_sum(window_size=window, by="time")
    )

    # logarithmic distribution Benford's law
    expected_freq = pl.Series([np.log(i + 1) - np.log(i) for i in range(1, 10)])

    df_first_digits = (
        df_first_digits
        # Calculate rolling count of trades for each leading digit
        .with_columns(
            df_first_digits.rolling(index_column="time", period=window)
            # Calculate observed frequences of whale buy orders grouped by leading digit
            .agg(
                [
                    (pl.col(col) * pl.col("is_whale") * pl.col("is_buy")).sum()
                    for col in digit_cols
                ]
            )
        )
        # Calculate rolling expected frequency for each digit
        .with_columns_seq(
            [
                (pl.col("rolling_num_whale_trades") * expected_freq[i - 1]).alias(
                    f"expected_freq_{i}"
                )
                for i in range(1, 10)
            ]
        )
    )

    expected_freq_cols = [f"expected_freq_{i}" for i in range(1, 10)]

    df_first_digits = df_first_digits.with_columns(
        chi2_stat=(
            (df_first_digits[digit_cols] - df_first_digits[expected_freq_cols])
            / df_first_digits[expected_freq_cols]
        )
        .sum_horizontal()
        .pow(2)
    )

    df_group = df_group.with_columns_seq(df_first_digits.select("chi2_stat"))

    # ---------------------------------------------------------------------------------------------------------------
    # --Output created features -------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------

    df_features = (
        df_group.lazy()
        .rolling(index_column="time", period="1d")
        .agg(
            rolling_whale_imbalance_ratio=pl.col("whale_imbalance_ratio").mean(),
            rolling_regular_imbalance_ratio=pl.col("regular_imbalance_ratio").mean(),
            *[pl.col(col).mean().alias(f"{col}_mean") for col in weekday_cols],
            *[pl.col(col).max().alias(f"{col}_max") for col in weekday_cols],
            *[pl.col(col).mean().alias(f"{col}_mean") for col in hour_cols],
            *[pl.col(col).max().alias(f"{col}_max") for col in hour_cols],
            *[
                pl.col("chi2_stat").mean().alias("chi2_stat_mean"),
                pl.col("chi2_stat").max().alias("chi2_stat_max"),
            ],
        )
    ).collect()

    df_features: pd.DataFrame = df_features.filter(
        (pl.col("time") <= pump_event.pump_time)
        & (pl.col("time") >= pump_event.pump_time - timedelta(hours=24 * 14))
    ).to_pandas()

    df_features = df_features.resample(
        on="time",
        rule=window_output,
        offset=timedelta(
            minutes=pump_event.pump_time.minute, seconds=pump_event.pump_time.second
        ),
    ).first()

    df_features = df_features.reset_index(drop=True)

    df_features.index = [f"{i}_H" for i in range(1, 24 * 14 + 1)]
    df_features = df_features.unstack().to_frame().sort_index(level=1).T
    df_features.columns = df_features.columns.map("_".join)

    return df_features
