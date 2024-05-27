from typing import *
from datetime import timedelta, datetime
from pipes.dataloader import PumpEvent
from tsfresh.feature_extraction.feature_calculators import benford_correlation
from sklearn.metrics import auc
from scipy.stats import powerlaw


import pandas as pd
import numpy as np
import os
import warnings


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


def transform_to_features_30(
    df_ticker: pd.DataFrame, pump: PumpEvent, df_cmc_ticker: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    
    all_features: Dict[str, float] = {}

    df_ticker["quote_abs"] = df_ticker["price"] * df_ticker["qty"]

    df_ticker["qty_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["qty"]
    df_ticker["quote_sign"] = (1 - 2 * df_ticker["isBuyerMaker"]) * df_ticker["quote_abs"]

    df_trades: pd.DataFrame = df_ticker.groupby("time").agg(
        price_first=("price", "first"),
        price_last=("price", "last"),
        price_max=("price", "max"),
        price_min=("price", "min"),
        qty_sign=("qty_sign", "sum"),
        qty_abs=("qty", "sum"),
        quote_sign=("quote_sign", "sum"),
        quote_abs=("quote_abs", "sum"),
    )

    df_trades["is_long"] = df_trades["qty_sign"] >= 0
    df_trades["quote_long"] = df_trades["quote_abs"] * df_trades["is_long"]
    df_trades = df_trades.reset_index()

    df_trades["quote_slippage_abs"] = (
        df_trades["quote_sign"] - df_trades["qty_sign"] * df_trades["price_first"]
    )
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])

    time_ub: pd.Timestamp = pump.time.floor("1h") - timedelta(hours=1)

    df_trades_1d: pd.DataFrame = df_trades[
        df_trades["time"] >= time_ub - timedelta(days=1)
    ].copy()

    # ------------------------------------------------------------------------------------------------------------------------------------
    # AUC Features
    # ------------------------------------------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based Log returns features
    # ------------------------------------------------------------------------------------------------------------------------------------
    df_candles_1h["imbalance_ratio"] = df_candles_1h["quote_sign"] / df_candles_1h["quote_abs"]
    df_candles_1h["long_trades_ratio"] = df_candles_1h["num_long_trades"] / df_candles_1h["num_trades"]
    df_candles_1h["quote_slippage_imbalance_ratio"] = df_candles_1h["quote_slippage_sign"] / df_candles_1h["quote_slippage_abs"]

    # calculate log returns
    df_candles_1h["close"] = df_candles_1h["close"].ffill()
    df_candles_1h["log_returns"] = np.log(df_candles_1h["close"] / df_candles_1h["close"].shift(1))
    df_candles_1h = df_candles_1h.iloc[1:]
    df_candles_1h = df_candles_1h.reset_index()

    df_30d: pd.DataFrame = df_candles_1h[
        df_candles_1h["time"] >= time_ub - timedelta(days=30)
    ].copy()

    offsets: List[timedelta] = [
        timedelta(days=i) for i in [1, 3, 7, 14]
    ]

    labels: List[str] = [
        f"{i}d" for i in [1, 3, 7, 14]
    ]

    hourly_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval = df_candles_1h[
            df_candles_1h["time"] >= time_ub - offset
        ]

        # Share of long trades scaled by long run moments 
        hourly_features[f"long_trades_1h_ratio_zscore_{label}"] = (
            df_interval["long_trades_ratio"].mean() - df_30d["long_trades_ratio"].mean()
        ) / df_30d["long_trades_ratio"].std()
        hourly_features[f"long_trades_1h_ratio_overall_{label}"] = df_interval["num_long_trades"].sum() / df_interval["num_trades"].sum()
        
        # Imbalance ratio features
        hourly_features[f"imbalance_ratio_1h_mean_{label}"] = df_interval["imbalance_ratio"].mean()
        hourly_features[f"imbalance_ratio_1h_std_{label}"] = df_interval["imbalance_ratio"].std()

        # Log returns features
        hourly_features[f"log_returns_1h_zscore_{label}"] = df_interval["log_returns"].mean() / df_30d["log_returns"].std()
        hourly_features[f"log_returns_1h_std_{label}"] = df_interval["log_returns"].std()

        hourly_features[f"quote_abs_1h_zscore_{label}"] = (
            df_interval["quote_abs"].mean() - df_30d["quote_abs"].mean()
        ) / df_30d["quote_abs"].std()

        # Quote slippage features
        hourly_features[f"quote_slippage_imbalance_ratio_1h_mean_{label}"] = df_interval["quote_slippage_imbalance_ratio"].mean()
        hourly_features[f"quote_slippage_imbalance_ratio_1h_std_{label}"] = df_interval["quote_slippage_imbalance_ratio"].std()
        # find slippage imbalance ratio over the whole interval
        hourly_features[f"quote_slippage_imbalance_ratio_1h_overall_{label}"] = (
            df_interval["quote_slippage_sign"].sum() / df_interval["quote_slippage_abs"].sum()
        )
        # Share of slippage in the whole volume
        hourly_features[f"quote_slippage_1h_quote_abs_ratio_{label}"] = df_interval["quote_slippage_abs"].sum() / df_interval["quote_abs"].sum()

    all_features.update(hourly_features)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based Powerlaw features
    # ------------------------------------------------------------------------------------------------------------------------------------
    powerlaw_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[
            (df_trades["time"] >= time_ub - offset)
        ]

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

    all_features.update(powerlaw_features)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based Benford's correlation. TSFresh
    # ------------------------------------------------------------------------------------------------------------------------------------
    benford_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[
            (df_trades["time"] >= pump.time.floor("1h") - offset)
        ]

        # Benford correlation
        benford_features[f"benford_law_correlation_{label}"] = benford_correlation(df_interval["qty_abs"])

    all_features.update(benford_features)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based. Quote features / Quantiles ratios
    # ------------------------------------------------------------------------------------------------------------------------------------
    quote_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades[
            (df_trades["time"] >= time_ub - offset)
        ]
        df_long: pd.DataFrame = df_interval[df_interval["is_long"]]
        df_short: pd.DataFrame = df_interval[~df_interval["is_long"]]

        # 999 quantiles
        long_whale_quantile: float = df_long["quote_abs"].quantile(.999)
        short_whale_quantile: float = df_short["quote_abs"].quantile(.999)

        # 99 quantiles
        long_99_quantile: float = df_long["quote_abs"].quantile(.99)
        short_99_quantile: float = df_short["quote_abs"].quantile(.99)

        # 95 quantiles
        long_95_quantile: float = df_long["quote_abs"].quantile(.95)
        short_95_quantile: float = df_short["quote_abs"].quantile(.95)

        long_median_quote: float = df_long["quote_abs"].median()
        short_nedian_quote: float = df_short["quote_abs"].median()

        quote_features[f"long_whale_99_ratio_{label}"] = long_whale_quantile / long_99_quantile # how spread out long quantiles become
        quote_features[f"short_whale_99_ratio_{label}"] = short_whale_quantile / short_99_quantile
        quote_features[f"long_whale_short_whale_ratio_{label}"] = long_whale_quantile / short_whale_quantile # how extended is long quantile compared to short

        quote_features[f"long_99_long_95_ratio_{label}"] = long_99_quantile / long_95_quantile
        quote_features[f"short_99_95_ratio_{label}"] = short_99_quantile / short_95_quantile
        quote_features[f"long_99_short_99_ratio_{label}"] = long_99_quantile / short_99_quantile

        quote_features[f"long_whale_median_ratio_{label}"] = long_whale_quantile / long_median_quote
        quote_features[f"short_whale_median_ratio_{label}"] = short_whale_quantile / short_nedian_quote

    all_features.update(quote_features)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based. Empty trading minutes share
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # rebin data into 1min candles
    df_candles_1min: pd.DataFrame = df_trades.resample(on="time", rule="1min", label="left", closed="left").agg(
        open=("price_first", "first"),
        close=("price_last", "last"),
        low=("price_min", "min"),
        high=("price_max", "max"),
        num_trades=("quote_abs", "count"),
        num_long_trades=("is_long", "sum"),
        quote_abs=("quote_abs", "sum"),
        quote_sign=("quote_sign", "sum"),
        quote_slippage_abs=("quote_slippage_abs", "sum"),
        quote_slippage_sign=("quote_slippage_sign", "sum")
    ).reset_index()

    empty_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_candles_1min[
            (df_candles_1min["time"] >= time_ub - offset)
        ]

        empty_features[f"empty_trading_minutes_ratio_{label}"] = df_interval[df_interval["num_trades"] == 0].shape[0] / df_interval.shape[0]

    all_features.update(empty_features)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # Interval-based. Short run features
    # ------------------------------------------------------------------------------------------------------------------------------------
    offsets: List[timedelta] = [
        timedelta(hours=i) for i in [1, 3, 7, 12]
    ]

    labels: List[str] = [
        f"{i}h" for i in [1, 3, 7, 12]
    ]

    short_features: Dict[str, float] = {}

    for offset, label in zip(offsets, labels):
        df_interval: pd.DataFrame = df_trades_1d[
            df_trades_1d["time"] >= time_ub - offset
        ]

        df_long: pd.DataFrame = df_interval[df_interval["is_long"]]
        df_short: pd.DataFrame = df_interval[~df_interval["is_long"]]

        # Share of long trades within the time interval and the whole day 
        short_features[f"long_trades_ratio_{label}"] = df_interval["is_long"].sum() / df_interval.shape[0]
        short_features[f"long_trades_ratio_{label}_1d"] = df_interval["is_long"].sum() / df_trades_1d.shape[0]
        # Share of volume within the time interval and the whole day
        short_features[f"quote_abs_ratio_{label}_1d"] = df_interval["quote_abs"].sum() / df_trades_1d["quote_abs"].sum()
        short_features[f"quote_abs_long_ratio_{label}_1d"] = df_long["quote_abs"].sum() / df_interval["quote_abs"].sum()
        # Imbalance ratio
        short_features[f"imbalance_ratio_{label}"] = df_interval["quote_sign"].sum() / df_interval["quote_abs"].sum()
        # Quote slippage features
        short_features[f"quote_slippage_ratio_{label}_1d"] = df_interval["quote_slippage_abs"].sum() / df_trades_1d["quote_slippage_abs"].sum() # share of overall slippage
        short_features[f"quote_slippage_imbalance_ratio_{label}"] = df_interval["quote_slippage_sign"].sum() / df_interval["quote_slippage_abs"].sum()
        # Share of slippage quote compared to volume
        short_features[f"quote_slippage_quote_abs_ratio_{label}"] = df_interval["quote_slippage_abs"].sum() / df_interval["quote_abs"].sum()

    all_features.update(short_features)

    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features








