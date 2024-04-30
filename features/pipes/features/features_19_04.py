from typing import *
from datetime import timedelta, datetime
from pipes.dataloader import PumpEvent
from tsfresh import extract_features
from statsmodels.tsa.stattools import adfuller
from tsfresh.feature_extraction.feature_calculators import cid_ce
from tsfresh.feature_extraction.feature_calculators import benford_correlation
from scipy.stats import powerlaw

import pandas as pd
import numpy as np
import os


def check_num_times_has_been_pumped(pump: PumpEvent, ticker: str) -> int:
    """log_Returns a number of time ticker has been pumped before this pump event"""
    df_pumps: pd.DataFrame = pd.read_json("data/pumps/pumps_31_03_2024.json")
    df_pumps["time"] = pd.to_datetime(df_pumps["time"])

    df_prev_pumps: pd.DataFrame = df_pumps[
        (df_pumps["ticker"] == ticker) & (df_pumps["time"] < pump.time)
    ].copy()

    return df_prev_pumps.shape[0]





def transform_to_features_19(
    df_ticker: pd.DataFrame, pump: PumpEvent, df_cmc_ticker: pd.DataFrame, ticker: str
) -> pd.DataFrame:

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
        # Add BTC slippage
    )

    features: Dict[str, Any] = {}

    df_trades["is_long"] = df_trades["qty_sign"] >= 0
    df_trades = df_trades.reset_index()

    df_candles_1h: pd.DataFrame = df_trades.resample(
        on="time", rule="1h", label="left", closed="left"
    ).agg(
        open=("price_first", "first"),
        close=("price_last", "last"),
        vol=("qty_abs", "sum"),
        net_position=("qty_sign", "sum"),
        net_quote=("quote_sign", "sum"),
        quote_vol=("quote_abs", "sum"),
        num_trades=("price_first", "count"),
        num_long_trades=("is_long", "sum")
    )

    df_candles_1h["close"] = df_candles_1h["close"].ffill()
    df_candles_1h["log_return"] = np.log(df_candles_1h["close"] / df_candles_1h["close"].shift(1))
    df_candles_1h = df_candles_1h.iloc[1:]
    df_candles_1h = df_candles_1h.reset_index()

    offsets: List[timedelta] = [
        timedelta(days=i) for i in [1, 2, 5, 7, 14, 21, 30]
    ]

    labels: List[str] = [
        "1d", "2d", "5d", "7d", "14d", "21d", "30d"
    ]

    df_population: pd.DataFrame = df_candles_1h[
        df_candles_1h["time"] >= pump.time.floor("1h") - timedelta(days=30)
    ].copy()

    hourly_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_candles_1h[
            (df_candles_1h["time"] >= pump.time.floor("1h") - offset)
        ].copy()

        # Log return features
        hourly_features[f"log_return_1h_z_score_{label}"] = df_interval["log_return"].mean() / df_population["log_return"].std()
        hourly_features[f"log_return_1h_std_{label}"] = df_interval["log_return"].std()
        # Num trades features
        hourly_features[f"z_score_num_trades_1h_{label}"] = df_interval["num_trades"].mean() / df_population["num_trades"].std()
        hourly_features[f"share_empty_trading_hours_{label}"] = df_interval[df_interval["num_trades"] == 0].shape[0] / df_interval.shape[0]
        # Imbalance ratio
        hourly_features[f"imbalance_ratio_1h_{label}"] = df_interval["net_quote"].sum() / df_interval["quote_vol"].sum()

    features.update(hourly_features)

    powerlaw_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[
            (df_trades["time"] >= pump.time.floor("1h") - offset)
        ].copy()

        # Compute alpha of powerlaw distribution for quote_abs, quote_long, quote_short
        powerlaw_features[f"quote_abs_powerlaw_alpha_{label}"] = powerlaw.fit(df_interval["quote_abs"])[0]
        # alpha for long trades
        powerlaw_features[f"long_quote_abs_powerlaw_alpha_{label}"] = powerlaw.fit(
            df_interval[df_interval["is_long"]]["quote_abs"]
        )[0]
        # alpha for short trades
        powerlaw_features[f"short_quote_abs_powerlaw_alpha_{label}"] = powerlaw.fit(
            df_interval[~df_interval["is_long"]]["quote_abs"]
        )[0]

    features.update(powerlaw_features)

    benford_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[
            (df_trades["time"] >= pump.time.floor("1h") - offset)
        ].copy()

        # Benford correlation
        benford_features[f"benford_law_correlation_{label}"] = benford_correlation(df_interval["qty_abs"])

    features.update(benford_features)

    quote_qty_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[df_trades["time"] >= pump.time.floor("1h") - offset].copy()
        
        long_whale_quantile_999 = df_interval[df_interval["is_long"]]["quote_abs"].quantile(.999)
        short_whale_quantile_999 = df_interval[~df_interval["is_long"]]["quote_abs"].quantile(.999)
        whale_quantile_999 = df_interval["quote_abs"].quantile(.999)
        
        long_whale_quantile_99 = df_interval[df_interval["is_long"]]["quote_abs"].quantile(.99)
        short_whale_quantile_99 = df_interval[~df_interval["is_long"]]["quote_abs"].quantile(.99)
        whale_quantile_999 = df_interval["quote_abs"].quantile(.99)

        long_whale_quantile_95 = df_interval[df_interval["is_long"]]["quote_abs"].quantile(.95)
        short_whale_quantile_95 = df_interval[~df_interval["is_long"]]["quote_abs"].quantile(.95)
        whale_quantile_999 = df_interval["quote_abs"].quantile(.99)


        quote_qty_features[f"long_overall_quantile_999_ratio_{label}"] = long_whale_quantile_999 / whale_quantile_999
        quote_qty_features[f"short_overall_quantile_999_ratio_{label}"] = short_whale_quantile_999 / whale_quantile_999

        quote_qty_features[f"long_quantile_99_999_ratio_{label}"] = long_whale_quantile_999 / long_whale_quantile_99
        quote_qty_features[f"short_quantile_99_999_ratio_{label}"] = short_whale_quantile_999 / short_whale_quantile_99

        quote_qty_features[f"long_quantile_95_99_ratio_{label}"] = long_whale_quantile_95 / long_whale_quantile_99
        quote_qty_features[f"short_quantile_95_99_ratio_{label}"] = short_whale_quantile_95 / short_whale_quantile_99

    features.update(quote_qty_features)

    df_trades_1d: pd.DataFrame = df_trades[
        df_trades["time"] >= pump.time.floor("1h") - timedelta(days=1)
    ].copy()

    df_trades_1d["quote_sign_quantile_999"] = df_trades_1d.rolling(on="time", window="1h")["quote_sign"].quantile(.999)

    hour_bins: List[int] = [1, 4, 8, 12]

    offsets: List[timedelta] = [
        timedelta(hours=i) for i in hour_bins
    ]

    labels: List[str] = [
        f"{i}h" for i in hour_bins
    ]

    df_candles_1h_1d: pd.DataFrame = df_candles_1h[
        df_candles_1h["time"] >= pump.time.floor("1h") - timedelta(days=1)
    ].copy()

    num_trades_1d: int = df_candles_1h_1d["num_trades"].sum()
    quote_vol_1d: int = df_candles_1h_1d["quote_vol"].sum()
    num_long_trades_1d: int = df_candles_1h_1d["num_long_trades"].sum()

    short_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_candles_1h[df_candles_1h["time"] >= pump.time.floor("1h") - offset].copy()
        short_features[f"num_trades_{label}_1d_ratio"] = df_interval["num_trades"].sum() / num_trades_1d
        short_features[f"quote_vol_{label}_1d_ratio"] = df_interval["quote_vol"].sum() / quote_vol_1d
        short_features[f"long_num_trades_{label}_1d_ratio"] = df_interval["num_long_trades"].sum() / num_long_trades_1d

    features.update(short_features)

    df_features = pd.DataFrame(data=features.values()).T
    df_features.columns = features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features


