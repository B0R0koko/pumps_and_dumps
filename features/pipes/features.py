from typing import *
from datetime import timedelta
from pipes.dataloader import PumpEvent

import pandas as pd
import numpy as np
import os


def check_num_times_has_been_pumped(pump: PumpEvent) -> int:
    """log_Returns a number of time ticker has been pumped before this pump event"""
    df_pumps: pd.DataFrame = pd.read_json("data/pumps/pumps_31_03_2024.json")
    df_pumps["time"] = pd.to_datetime(df_pumps["time"])

    df_prev_pumps: pd.DataFrame = df_pumps[
        (df_pumps["ticker"] == pump.ticker) & (df_pumps["time"] < pump.time)
    ].copy()

    return df_prev_pumps.shape[0]


def transform_to_simple_features(
    df_ticker: pd.DataFrame, pump: PumpEvent, df_cmc_ticker: pd.DataFrame, ticker: str
) -> pd.DataFrame:

    df_ticker["quote"] = df_ticker["price"] * df_ticker["qty"]
    df_ticker["qty_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["qty"]
    df_ticker["quote_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["quote"]

    df_trades: pd.DataFrame = df_ticker.groupby("time").agg(
        price_first=("price", "first"),
        price_last=("price", "last"),
        qty_sign=("qty_sign", "sum"),
        qty_abs=("qty", "sum"),
        quote_sign=("quote_sign", "sum"),
        quote_abs=("quote", "sum"),
    )
    df_trades = df_trades.reset_index()

    all_features: Dict[str, Any] = {}

    # -----------------------------------------------------------------------------------------------
    # Imbalance ratio features
    # -----------------------------------------------------------------------------------------------

    window_sizes: List[timedelta] = [timedelta(days=days) for days in [1, 7, 14, 30]]
    window_names: List[str] = ["1d", "7d", "14d", "30d"]

    imbalance_features = {}

    for window_size, window_name in zip(window_sizes, window_names):
        df_trades_window: pd.DataFrame = df_trades[
            df_trades["time"] >= pump.time.round("1h") - timedelta(hours=1) - window_size
        ]
        imbalance_ratio: float = (
            df_trades_window["qty_sign"].sum() / df_trades_window["qty_abs"].sum()
        )
        imbalance_features[f"imbalance_ratio_{window_name}"] = imbalance_ratio

    all_features.update(imbalance_features)

    # -----------------------------------------------------------------------------------------------
    # 1 hour binned features (log returns, num_trades, quote_asset_volume)
    # -----------------------------------------------------------------------------------------------

    df_trades_1h: pd.DataFrame = df_trades.resample(
        rule="1h", on="time", label="right", closed="right"
    ).agg(
        qty_abs_1h=("qty_abs", "sum"),
        price_first=("price_first", "first"),
        price_last=("price_last", "last"),
        num_trades_1h=("qty_sign", "count"),
        quote_abs_1h=("quote_abs", "sum"),
    )

    # calculate log returns
    df_trades_1h["price_last"] = df_trades_1h["price_last"].ffill()
    df_trades_1h["log_returns_1h"] = np.log(
        df_trades_1h["price_last"] / df_trades_1h["price_last"].shift(1)
    )
    df_trades_1h = df_trades_1h[df_trades_1h["log_returns_1h"].notna()]
    df_trades_1h = df_trades_1h.reset_index()

    features_binned_1h = {}

    for window_size, window_name in zip(window_sizes, window_names):
        df_window: pd.DataFrame = df_trades_1h[
            df_trades_1h["time"] >= pump.time.round("1h") - timedelta(hours=1) - window_size
        ]
        for col in ["log_returns_1h", "num_trades_1h", "quote_abs_1h"]:
            # Add features
            features_binned_1h[f"{col}_mean_{window_name}"] = df_window[col].mean()
            features_binned_1h[f"{col}_std_{window_name}"] = df_window[col].std()

    all_features.update(features_binned_1h)

    all_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump)

    # Exchange volume ratio
    daily_exchange_vol_features = {}

    df_trades["date"] = df_trades["time"].dt.floor("1d")

    df_vol = df_trades.groupby("date")["quote_abs"].sum().to_frame().reset_index()
    df_vol = df_vol[
        df_vol["date"] < pump.time.floor("1d")
    ].copy()

    df_vol: pd.DataFrame = df_vol.merge(
        df_cmc_ticker[["snapshot", "trading_volume_btc"]], left_on="date", right_on="snapshot", how="left"
    )

    for window_size, window_name in zip(window_sizes, window_names):
        df_window: pd.DataFrame = df_vol[
            df_vol["date"] >= pump.time.floor("1d") - window_size
        ].copy()

        df_window["daily_exchange_vol_ratio"] = df_window["quote_abs"] / df_window["trading_volume_btc"]
        daily_exchange_vol_features[f"daily_exchange_vol_ratio_{window_name}_mean"] = (
            df_window["daily_exchange_vol_ratio"].mean() 
        )
        daily_exchange_vol_features[f"daily_exchange_vol_ratio_{window_name}_std"] = (
            df_window["daily_exchange_vol_ratio"].std() 
        )

    all_features.update(daily_exchange_vol_features)

    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    return df_features
