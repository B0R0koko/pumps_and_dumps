from typing import *
from datetime import timedelta
from pipes.dataloader import PumpEvent

import polars as pl
import pandas as pd
import numpy as np


def transform_features(
    df: pl.DataFrame, pump: PumpEvent, window: str, window_output: str
) -> pd.DataFrame:

    # Aggregate into so-called rush trades in the literature

    df_trades = (
        df.with_columns(
            qty_sign=(1 - 2 * pl.col("isBuyerMaker").cast(pl.Int8)) * pl.col("qty"),
        )
        .group_by("time", maintain_order=True)
        .agg(
            # Price aggregation
            price_first=pl.col("price").first(),
            price_last=pl.col("price").last(),
            price_mean=pl.col("price").mean(),
            price_min=pl.col("price").min(),
            price_max=pl.col("price").max(),
            # Check if the agg trade corresponds to one person
            isBuyerMaker_mean=pl.col("isBuyerMaker").mean(),
            # Qty aggregation
            qty_sign=pl.col("qty_sign").sum(),
            qty_abs=pl.col("qty").sum(),
            # Market slippage metrics (liquidity metrics)
            btc_lost_to_slippage=(
                # actual quote paid - what could have been achieved with limit order
                (pl.col("qty_sign") * pl.col("price")).sum()
                - (pl.col("qty_sign").sum() * pl.col("price").first())
            ),
            btc_quote_spent=((pl.col("qty_sign") * pl.col("price")).sum().abs()),
        )
        # Remove panic! cases when isBuyerMaker_mean \in (0, 1)
        # .filter(
        #     (pl.col("isBuyerMaker_mean") == 0) | (pl.col("isBuyerMaker_mean") == 1)
        # )
        .with_columns(time=pl.col("time").set_sorted())
    )

    df_trades = df_trades.filter(
        (pl.col("time") >= pump.pump_time - timedelta(days=21))
        & (pl.col("time") <= pump.pump_time)
    )

    # Calculate slippages
    df_trades = (
        df_trades.with_columns(is_long=pl.col("qty_sign") >= 0)
        .with_columns(
            btc_lost_by_buyers=pl.col("btc_lost_to_slippage") * pl.col("is_long"),
            btc_lost_by_sellers=pl.col("btc_lost_to_slippage") * ~pl.col("is_long"),
        )
        .with_columns(
            cum_btc_lost_by_buyers=pl.col("btc_lost_by_buyers").cum_sum(),
            cum_btc_lost_by_sellers=pl.col("btc_lost_by_sellers").cum_sum(),
        )
    )

    # Define offset for time grouping
    offset = timedelta(minutes=pump.pump_time.minute, seconds=pump.pump_time.second)

    # Find whale long short quantiles
    long_quantile = (
        df_trades.filter(pl.col("is_long"))
        .group_by_dynamic(
            index_column="time", every="1h", period="7d", label="right", offset=offset
        )
        .agg(
            long_quantile_whale=pl.col("qty_abs").quantile(0.999),
            long_quantile_99=pl.col("qty_abs").quantile(0.99),
            long_quantile_95=pl.col("qty_abs").quantile(0.95),
        )
    )

    df_quantiles = (
        df_trades.filter(~pl.col("is_long"))
        .group_by_dynamic(
            index_column="time", every="1h", period="7d", label="right", offset=offset
        )
        .agg(
            short_quantile_whale=pl.col("qty_abs").quantile(0.999),
            short_quantile_99=pl.col("qty_abs").quantile(0.99),
            short_quantile_95=pl.col("qty_abs").quantile(0.95),
        )
        .join(long_quantile, how="left", on="time")
    )

    # Merge this quantiles to the df_trades dataframe
    df_trades = df_trades.with_columns(
        time_quantile=(
            (pl.col("time") + timedelta(hours=1)).dt.truncate("1h") + offset
        ).cast(pl.Datetime(time_unit="ns"))
    ).join(df_quantiles, left_on="time_quantile", right_on="time", how="left")

    # add is_whale and is_99 indicators
    df_trades = df_trades.with_columns(
        # Whales
        is_long_whale=(
            (pl.col("qty_abs") >= pl.col("long_quantile_whale")) & pl.col("is_long")
        ),
        is_short_whale=(
            (pl.col("qty_abs") >= pl.col("short_quantile_whale")) & ~pl.col("is_long")
        ),
        # (0.99; 0.999)
        is_long_99=(
            pl.col("qty_abs").is_between(
                pl.col("long_quantile_99"), pl.col("long_quantile_whale")
            )
            & pl.col("is_long")
        ),
        is_short_99=(
            pl.col("qty_abs").is_between(
                pl.col("short_quantile_99"), pl.col("short_quantile_whale")
            )
            & ~pl.col("is_long")
        ),
    ).with_columns(
        is_whale=pl.col("is_long_whale") | pl.col("is_short_whale"),
        is_99=pl.col("is_long_99") | pl.col("is_short_99"),
    )

    # Calculate slippages by quantiles
    df_slippages = (
        df_trades.group_by_dynamic(
            index_column="time", every="1h", period="7d", label="right", offset=offset
        )
        .agg(
            # Whale slippage
            long_whale_slippage=(
                pl.col("btc_lost_to_slippage") * pl.col("is_long_whale")
            ).sum(),
            short_whale_slippage=(
                pl.col("btc_lost_to_slippage") * pl.col("is_short_whale")
            ).sum(),
            # 99 quantile slippage
            long_99_slippage=(
                pl.col("btc_lost_to_slippage") * pl.col("is_long_99")
            ).sum(),
            short_99_slippage=(
                pl.col("btc_lost_to_slippage") * pl.col("is_short_99")
            ).sum(),
        )
        .with_columns(
            net_whale_slippage=pl.col("long_whale_slippage")
            - pl.col("short_whale_slippage"),
            net_99_slippage=pl.col("long_99_slippage") - pl.col("short_99_slippage"),
        )
        .select("time", "net_whale_slippage", "net_99_slippage")
    )

    # Whale imbalance ratio
    df_imbalances = (
        df_trades.group_by_dynamic(
            index_column="time", every="1h", period="7d", label="right", offset=offset
        )
        .agg(
            # Whale quantities
            whale_net_position=(pl.col("qty_sign") * pl.col("is_whale")).sum(),
            whale_overall_vol=(pl.col("qty_abs") * pl.col("is_whale")).sum(),
            # 99 quantile
            quan_99_net_position=(pl.col("qty_sign") * pl.col("is_99")).sum(),
            quan_99_overall_vol=(pl.col("qty_abs") * pl.col("is_99")).sum(),
        )
        .with_columns(
            whale_imbalance_ratio=pl.col("whale_net_position")
            / pl.col("whale_overall_vol"),
            quan_99_imbalance_ratio=pl.col("quan_99_net_position")
            / pl.col("quan_99_overall_vol"),
        )
        .select("time", "whale_imbalance_ratio", "quan_99_imbalance_ratio")
    )

    # Number of trades by quantiles
    df_num_trades = (
        df_trades.group_by_dynamic(
            index_column="time", every="1h", period="7d", label="right", offset=offset
        )
        .agg(
            num_trades_long_whale=pl.col("is_long_whale").sum(),
            num_trades_short_whale=pl.col("is_short_whale").sum(),
            num_trades_long_99=pl.col("is_long_99").sum(),
            num_trades_short_99=pl.col("is_short_99").sum(),
        )
        .with_columns(
            long_whale_99_num_trades_ratio=(
                pl.col("num_trades_long_whale") / pl.col("num_trades_long_99")
            ),
            short_whale_99_num_trades_ratio=(
                pl.col("num_trades_short_whale") / pl.col("num_trades_short_99")
            ),
        )
        .select(
            "time", "long_whale_99_num_trades_ratio", "short_whale_99_num_trades_ratio"
        )
    )

    # ---------------------------------------------------------------------------------------------------
    # OUTPUT TO TRAIN DATAFRAME
    # ---------------------------------------------------------------------------------------------------

    df_features = (
        df_slippages.join(df_num_trades, on="time", how="left")
        .join(df_imbalances, on="time", how="left")
        .filter(
            (pl.col("time") >= pump.pump_time - timedelta(days=7))
            & (  # 7 days before the pump
                pl.col("time") <= pump.pump_time - timedelta(hours=1)
            )  # add all features 1 hour prior to the pump
        )
        .to_pandas()
        .drop(columns=["time"])
    )

    # flatten the matrix into the vector
    df_features.index = [f"{i}_H" for i in range(1, 24 * 7 + 1)]
    df_features = df_features.unstack().to_frame().sort_index(level=1).T
    df_features.columns = df_features.columns.map("_".join)

    return df_features
