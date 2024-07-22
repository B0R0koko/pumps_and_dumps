from typing import *
from datetime import timedelta, datetime
from pipes.dataloader import PumpEvent
from scipy.stats import powerlaw


import pandas as pd
import numpy as np
import os
import warnings
import gc


warnings.filterwarnings("ignore")



def check_num_times_has_been_pumped(pump: PumpEvent, ticker: str) -> int:
    """log_Returns a number of time ticker has been pumped before this pump event"""
    df_pumps: pd.DataFrame = pd.read_json("data/pumps/pumps_31_03_2024.json")
    df_pumps["time"] = pd.to_datetime(df_pumps["time"])

    df_prev_pumps: pd.DataFrame = df_pumps[
        (df_pumps["ticker"] == ticker) & 
        (df_pumps["time"] >= pump.time.floor("1h") - timedelta(hours=1) - timedelta(days=90)) &
        (df_pumps["time"] < pump.time.floor("1h") - timedelta(hours=1))
    ].copy()

    return df_prev_pumps.shape[0]



def transform_to_features(
    df_ticker: pd.DataFrame, pump: PumpEvent, df_cmc_ticker: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    
    all_features: Dict[str, float] = {}

    df_ticker["quote"] = df_ticker["price"] * df_ticker["qty"]

    df_ticker["qty_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["qty"]
    df_ticker["quote_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["quote"]


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

    df_trades["is_long"] = df_trades["qty_sign"] >= 0
    df_trades["quote_long"] = df_trades["quote_abs"] * df_trades["is_long"]
    df_trades["quote_short"] = df_trades["quote_abs"] * ~df_trades["is_long"]

    df_trades = df_trades.reset_index()

    df_trades["quote_slippage_abs"] = (
        df_trades["quote_sign"] - df_trades["qty_sign"] * df_trades["price_first"]
    )
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])

    # Long, short quote slippage
    df_trades["quote_slippage_abs_long"] = df_trades["quote_slippage_abs"] * df_trades["is_long"]
    df_trades["quote_slippage_abs_short"] = df_trades["quote_slippage_abs"] * ~df_trades["is_long"]


    time_ub: pd.Timestamp = pump.time.floor("1h") - timedelta(hours=1)
    df_trades = df_trades[df_trades["time"] <= time_ub].copy()

    # Interval based features

    hour_bins: List[int] = [1, 3, 12, 24, 36, 48, 60, 72]

    df_72h = df_trades[df_trades["time"] >= time_ub - timedelta(hours=72)].copy()

    features_3d: Dict[str, float] = {}

    for hour in hour_bins:
        df_interval: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(hours=hour)].copy()

        # Number of trades features
        long_short_trades_ratio: float = df_interval["is_long"].sum() / (~df_interval["is_long"]).sum()
        long_trades_share: float = df_interval["is_long"].sum() / df_72h["is_long"].sum() # share of long trades within this bin from 72hours
        short_trades_share: float = (~df_interval["is_long"]).sum() / (~df_72h["is_long"]).sum() # share of short trades within bin from 72hours

        # Volume features in quote assets
        long_quote_volume_share: float = df_interval["quote_long"].sum() / df_72h["quote_long"].sum() # share of long volume from 72 hours
        short_quote_volume_share: float = df_interval["quote_short"].sum() / df_72h["quote_short"].sum()

        # Imbalance ratio
        imbalance_ratio: float = df_interval["quote_sign"].sum() / df_interval["quote_abs"].sum()

        # Quote slippages
        quote_slippage_imbalance_ratio: float = df_interval["quote_slippage_sign"].sum() / df_interval["quote_slippage_abs"].sum()
        quote_slippage_share: float = df_interval["quote_slippage_abs"].sum() / df_72h["quote_slippage_abs"].sum()
        quote_slippage_long_short_ratio: float = (
            df_interval["quote_slippage_abs_long"].sum() / df_interval["quote_slippage_abs_short"].sum()
        )
        quote_slippage_long_share: float = df_interval["quote_slippage_abs_long"].sum() / df_72h["quote_slippage_abs_long"].sum()
        
        # Share of slippages in overall volume
        quote_slippage_quote_share: float = df_interval["quote_slippage_abs"].sum() / df_interval["quote_abs"].sum()

        # Add features
        features_3d.update({
            f"long_short_trades_ratio_{hour}h": long_short_trades_ratio,
            f"long_trades_share_{hour}h_72H": long_trades_share,
            f"short_trades_share_{hour}h_72H": short_trades_share,

            f"long_quote_volume_share_{hour}h_72H": long_quote_volume_share,
            f"short_quote_volume_share_{hour}h_72H": short_quote_volume_share,

            f"imbalance_ratio_{hour}h": imbalance_ratio,

            f"quote_slippage_imbalance_ratio_{hour}h": quote_slippage_imbalance_ratio,
            f"quote_slippage_share_{hour}h_72H": quote_slippage_share,
            f"quote_slippage_long_short_ratio_{hour}h": quote_slippage_long_short_ratio,
            f"quote_slippage_quote_share_{hour}h_72H": quote_slippage_quote_share,
            f"quote_slippage_long_{hour}h_72H": quote_slippage_long_share,
        })

    all_features.update(features_3d)
    
    del df_72h
    gc.collect()


    daily_bins: List[int] = [1, 2, 3, 5, 7, 14]
    features_powerlaw: Dict[str, float] = {}


    for days in daily_bins:
        df_interval: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(days=days)].copy()

        alpha_quote_abs: float = powerlaw.fit(df_interval["quote_abs"])[0]
        alpha_quote_abs_long: float = powerlaw.fit(df_interval["quote_long"])[0]
        alpha_quote_abs_short: float = powerlaw.fit(df_interval["quote_short"])[0]

        features_powerlaw.update({
            f"alpha_quote_abs_{days}d": alpha_quote_abs,
            f"alpha_quote_abs_long_{days}d": alpha_quote_abs_long,
            f"alpha_quote_abs_short_{days}d": alpha_quote_abs_short
        })

    all_features.update(features_powerlaw)

    
    df_candles_1h: pd.DataFrame = df_trades.resample(
        on="time", rule="1h", label="left", closed="left"
    ).agg(
        open=("price_first", "first"),
        close=("price_last", "last"),
        low=("price_min", "min"),
        high=("price_max", "max"),
        num_trades=("quote_abs", "count"),
        num_long_trades=("is_long", "sum"),
        quote_abs=("quote_abs", "sum"),
        quote_sign=("quote_sign", "sum"),
        quote_long=("quote_long", "sum"),
        quote_slippage_abs=("quote_slippage_abs", "sum"),
        quote_slippage_sign=("quote_slippage_sign", "sum")
    ).reset_index()

    df_candles_1h["close"] = df_candles_1h["close"].ffill()
    df_candles_1h["log_returns"] = np.log(df_candles_1h["close"] / df_candles_1h["close"].shift(1))
    df_candles_1h = df_candles_1h.iloc[1:]
    df_candles_1h = df_candles_1h.reset_index()

    hour_bins: List[int] = [1, 3, 12, 24, 36, 48, 60, 72]

    df_candles_1h_72H = df_candles_1h[
        df_candles_1h["time"] >= time_ub - timedelta(hours=72)
    ].copy()

    features_candles_1h: Dict[str, float] = {}

    for hour in hour_bins:
        df_interval: pd.DataFrame = df_candles_1h[df_candles_1h["time"] >= time_ub - timedelta(hours=hour)].copy()

        features_candles_1h[f"hourly_log_return_{hour}h_zscore_72H"] = df_interval["log_returns"].mean() / df_candles_1h_72H["log_returns"].std()
        features_candles_1h[f"hourly_log_return_{hour}h_std"] = df_interval["log_returns"].std()

        features_candles_1h[f"hourly_volume_{hour}h_std_72H_std"] = df_interval["quote_abs"].std() / df_candles_1h_72H["quote_abs"].std()

    all_features.update(features_candles_1h)

    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features
