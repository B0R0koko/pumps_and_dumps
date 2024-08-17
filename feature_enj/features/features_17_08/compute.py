from typing import *
from datetime import timedelta, datetime
from pipes.dataloader import PumpEvent
from features.features_17_08.features import *

import pandas as pd
import numpy as np
import os
import warnings
import gc


warnings.filterwarnings("ignore")


def check_num_times_has_been_pumped(pump: PumpEvent, ticker: str) -> int:
    """log_Returns a number of time ticker has been pumped before this pump event"""
    df_pumps: pd.DataFrame = pd.read_json("data/pumps/merged_pumps_15_08.json")
    df_pumps["time"] = pd.to_datetime(df_pumps["time"])

    df_prev_pumps: pd.DataFrame = df_pumps[
        (df_pumps["ticker"] == ticker)
        & (df_pumps["time"] >= pump.time.round("1h") - timedelta(hours=1) - timedelta(days=90))
        & (df_pumps["time"] < pump.time.round("1h") - timedelta(hours=1))
    ].copy()

    return df_prev_pumps.shape[0]


def transform_to_features(
    df_ticker: pd.DataFrame, pump: PumpEvent, df_cmc_ticker: pd.DataFrame, ticker: str
) -> pd.DataFrame:

    all_features: Dict[str, float] = {}

    df_ticker["quote"] = df_ticker["qty"] * df_ticker["price"]
    df_ticker["qty_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["qty"]
    df_ticker["quote_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["quote"]

    # Aggregate by time into rush orders
    df_trades: pd.DataFrame = df_ticker.groupby("time").agg(
        price_first=("price", "first"),
        price_last=("price", "last"),
        price_max=("price", "max"),
        price_min=("price", "min"),
        qty_sign=("qty_sign", "sum"),
        qty_abs=("qty", "sum"),
        quote_sign=("quote_sign", "sum"),
        quote_abs=("quote", "sum"),
    )

    df_trades["is_long"] = df_trades["qty_sign"] >= 0  # is buyer
    df_trades["quote_long"] = df_trades["quote_abs"] * df_trades["is_long"]  # quote volume for longs
    df_trades["quote_short"] = df_trades["quote_abs"] * ~df_trades["is_long"]  # quote volume for shorts

    df_trades = df_trades.reset_index()

    # calculate slippages
    df_trades["quote_slippage_abs"] = np.abs(df_trades["quote_abs"] - df_trades["qty_abs"] * df_trades["price_first"])
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])
    df_trades["quote_slippage_long"] = df_trades["quote_slippage_abs"] * df_trades["is_long"]

    df_hourly_candles: pd.DataFrame = (
        df_trades.resample(on="time", rule="1h", closed="left").agg(
            open=("price_first", "first"),
            close=("price_last", "last"),
            low=("price_min", "min"),
            high=("price_max", "max"),
            volume_qty_abs=("qty_abs", "sum"),  # absolute volume in base asset
            volume_quote_abs=("quote_abs", "sum"),  # absolute volume in quote asset
            volume_quote_abs_long=("quote_long", "sum"),
            num_trades=("is_long", "count"),
            num_trades_long=("is_long", "sum"),
            quote_slippage_abs=("quote_slippage_abs", "sum"),  # slippage loss incurred by both buy and sell sides
            quote_slippage_abs_long=("quote_slippage_long", "sum"),  # quote slippage incurred by longs
        )
    ).reset_index()

    df_hourly_candles["log_return"] = np.log(df_hourly_candles["close"] / df_hourly_candles["close"].shift(1))

    time_ub: pd.Timestamp = pump.time.round("1h") - timedelta(hours=1)

    # long run mean and std of volumes in base and quote
    df_hourly_lr: pd.DataFrame = df_hourly_candles[
        df_hourly_candles["volume_quote_abs"] <= df_hourly_candles["volume_quote_abs"].quantile(0.99)
    ].copy()

    hour_offsets: List[int] = [1, 6, 24, 48, 72, 7 * 24, 14 * 24]

    hourly_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_hourly_candles[df_hourly_candles["time"] >= time_ub - timedelta(hours=offset)].copy()

        hourly_features[f"overall_return_{offset}h"] = calc_overall_return(df_window=df_window)
        # Scaled volumes in base and quote assets
        hourly_features[f"volume_quote_abs_zscore_{offset}h_30d"] = calc_volume_quote_abs_zscore(
            df_window=df_window, df_hourly_lr=df_hourly_lr
        )
        hourly_features[f"num_trades_long_share_{offset}h"] = calc_num_trades_long_share(df_window=df_window)
        hourly_features[f"volume_quote_long_share_{offset}h"] = calc_volume_quote_long_share(df_window=df_window)

        if offset == 1:
            continue
        # Hourly log returns volatility scaled by long run volatility
        hourly_features[f"log_return_std_{offset}h_30d"] = calc_log_return_std(
            df_window=df_window, df_hourly_lr=df_hourly_lr
        )
        # hourly log returns mean scaled by long run std -> z-score
        hourly_features[f"log_return_zscore_{offset}h_30d"] = calc_log_return_zscore(
            df_window=df_window, df_hourly_lr=df_hourly_lr
        )

    all_features.update(hourly_features)

    slippage_features: Dict[str, float] = {}

    df_hourly_candles_120h: pd.DataFrame = df_hourly_candles[
        df_hourly_candles["time"] >= time_ub - timedelta(hours=120)
    ].copy()

    hour_offsets: List[int] = [1, 3, 6, 12, 24, 48, 60]

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_hourly_candles[df_hourly_candles["time"] >= time_ub - timedelta(hours=offset)].copy()
        # Share of overall slippages of this time window in 120hours
        slippage_features[f"quote_slippage_abs_share_{offset}h_120h"] = calc_quote_slippage_abs_share(
            df_window=df_window, df_hourly_candles_120h=df_hourly_candles_120h
        )

    all_features.update(slippage_features)

    imbalance_features: Dict[str, float] = {}
    hour_offsets: List[int] = [1, 3, 6, 12, 24, 48, 60]

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()
        # Volume imbalance ratio to see if there is more buying pressure
        imbalance_features[f"quote_imbalance_ratio_{offset}h"] = df_window["quote_sign"].sum() / df_window["quote_abs"].sum()
        # Imbalance ratio in slippages to see if there is skew towards long slippages
        imbalance_features[f"quote_slippage_imbalance_ratio_{offset}h"] = (
            df_window["quote_slippage_sign"].sum() / df_window["quote_slippage_abs"].sum()
        )

    all_features.update(imbalance_features)

    evt_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()
        evt_features[f"quote_abs_powerlaw_alpha_{offset}h"] = calc_quote_abs_powerlaw_alpha(df_window=df_window)

    all_features.update(evt_features)

    # Output features
    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features
