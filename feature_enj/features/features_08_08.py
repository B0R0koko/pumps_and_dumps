from typing import *
from datetime import timedelta, datetime
from pipes.dataloader import PumpEvent
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


import pandas as pd
import numpy as np
import os
import warnings
import gc


warnings.filterwarnings("ignore")



def check_num_times_has_been_pumped(pump: PumpEvent, ticker: str) -> int:
    """log_Returns a number of time ticker has been pumped before this pump event"""
    df_pumps: pd.DataFrame = pd.read_json("data/pumps/pumps_27_05_2024.json")
    df_pumps["time"] = pd.to_datetime(df_pumps["time"])

    df_prev_pumps: pd.DataFrame = df_pumps[
        (df_pumps["ticker"] == ticker) & 
        (df_pumps["time"] >= pump.time.round("1h") - timedelta(hours=1) - timedelta(days=90)) &
        (df_pumps["time"] < pump.time.round("1h") - timedelta(hours=1))
    ].copy()

    return df_prev_pumps.shape[0]


def estimate_alpha_mle(data: pd.Series) -> float:
    """Hills estimator of powerlaw alpha"""
    n = len(data)
    xmin = min(data)
    alpha = 1 + n / np.sum(np.log(data / xmin))

    return alpha


def calculate_linear_liquidity_index(df_trades: pd.DataFrame) -> float:
    X = df_trades["quote_abs"]
    X = add_constant(X)

    Y = df_trades["quote_slippage_abs"]

    model = OLS(exog=X, endog=Y).fit()
    a, b = model.params

    return b


def regress_proba_slippage_on_quote_abs(df_trades: pd.DataFrame) -> Tuple[float, float]:
    """Compute the effect of quote_abs size on probability that the final trade will be executed with some slippage"""
    bins = np.arange(0, df_trades["quote_abs"].max(), 0.001)
    
    df_trades["quote_abs_bin"] = pd.cut(df_trades["quote_abs"], bins=bins)
    df_trades["quote_abs_bin"] = df_trades["quote_abs_bin"].apply(lambda x: x.right)

    df_trades["has_slippage"] = df_trades["quote_slippage_abs"] > 0 

    # Compute probas for all bins
    probas: List[dict] = []

    for bin, df_bin in df_trades.groupby("quote_abs_bin", observed=True):

        if df_bin.shape[0] >= 10:
            probas.append({
                "quote_bin": bin,
                "slippage_proba": df_bin["has_slippage"].mean() # compute the share of trades with any slippage within this bin
            })

    df_probas = pd.DataFrame(probas)

    # Regress
    X = df_probas["quote_bin"]
    X = add_constant(X)
    Y = df_probas["slippage_proba"]

    model = OLS(endog=Y, exog=X).fit()
    a, b = model.params

    return a, b


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

    df_trades["is_long"] = df_trades["qty_sign"] >= 0 # is buyer
    df_trades["quote_long"] = df_trades["quote_abs"] * df_trades["is_long"] # quote volume for longs
    df_trades["quote_short"] = df_trades["quote_abs"] * ~df_trades["is_long"] # quote volume for shorts
    
    df_trades = df_trades.reset_index()

    # calculate slippages
    df_trades["quote_slippage_abs"] = np.abs(
        df_trades["quote_abs"] - df_trades["qty_abs"] * df_trades["price_first"]
    )
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])
    df_trades["quote_slippage_long"] = df_trades["quote_slippage_abs"] * df_trades["is_long"]

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Hourly features. Log returns. Volumes
    # -------------------------------------------------------------------------------------------------------------------------------------------

    # Hourly candlesticks
    df_hourly_candles: pd.DataFrame = (
        df_trades
        .resample(on="time", rule="1h", closed="left")
        .agg(
            open=("price_first", "first"),
            close=("price_last", "last"),
            low=("price_min", "min"),
            high=("price_max", "max"),
            volume_qty_abs=("qty_abs", "sum"), # absolute volume in base asset
            volume_quote_abs=("quote_abs", "sum"), # absolute volume in quote asset
            volume_quote_abs_long=("quote_long", "sum"),
            num_trades=("is_long", "count"), 
            num_trades_long=("is_long", "sum"), 
            quote_slippage_abs=("quote_slippage_abs", "sum"), # slippage loss incurred by both buy and sell sides
            quote_slippage_abs_long=("quote_slippage_long", "sum") # quote slippage incurred by longs
        )
    ).reset_index()

    df_hourly_candles["log_return"] = np.log(
        df_hourly_candles["close"] / df_hourly_candles["close"].shift(1)
    )

    # Time upper boundary
    time_ub: pd.Timestamp = pump.time.round("1h") - timedelta(hours=1)

    # long run mean and std of volumes in base and quote
    df_hourly_lr: pd.DataFrame = df_hourly_candles[
        df_hourly_candles["volume_quote_abs"] <= df_hourly_candles["volume_quote_abs"].quantile(.99)
    ].copy()

    hour_offsets: List[int] = [1, 6, 24, 48, 72, 7*24, 14*24]

    hourly_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_hourly_candles[df_hourly_candles["time"] >= time_ub - timedelta(hours=offset)].copy()
        
        hourly_features[f"overall_return_{offset}h"] = (df_window["log_return"] + 1).prod() # overall return if held for the whole window up to the last hour

        # Scaled volumes in base and quote assets
        hourly_features[f"volume_quote_abs_zscore_{offset}h_30d"] = (
            (df_window["volume_quote_abs"].mean() - df_hourly_lr["volume_quote_abs"].mean()) / df_hourly_lr["volume_quote_abs"].std()
        )

        # hourly_features[f"num_trades_long_share_{offset}h"] = df_window["num_trades_long"].sum() / df_window["num_trades"].sum()
        hourly_features[f"volume_quote_long_share_{offset}h"] = df_window["volume_quote_abs_long"].sum() / df_window["volume_quote_abs"].sum()

        if offset == 1:
            continue
        # Hourly log returns volatility scaled by long run volatility
        hourly_features[f"log_return_std_{offset}h_30d"] = np.log(df_window["log_return"].std() / df_hourly_lr["log_return"].std())
        # hourly log returns mean scaled by long run std -> z-score
        hourly_features[f"log_return_zscore_{offset}h_30d"] = df_window["log_return"].mean() / df_hourly_lr["log_return"].std() 

    all_features.update(hourly_features)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Slippage features
    # -------------------------------------------------------------------------------------------------------------------------------------------

    slippage_features: Dict[str, float] = {}

    df_hourly_candles_120h: pd.DataFrame = df_hourly_candles[
        df_hourly_candles["time"] >= time_ub - timedelta(hours=120)
    ].copy()

    hour_offsets: List[int] = [1, 6, 24, 48, 72]

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_hourly_candles[df_hourly_candles["time"] >= time_ub - timedelta(hours=offset)].copy()
        # Share of overall slippages of this time window in 120hours
        slippage_features[f"quote_slippage_abs_share_{offset}h_120h"] = (
            df_window["quote_slippage_abs"].sum() / df_hourly_candles_120h["quote_slippage_abs"].sum()
        )

    all_features.update(slippage_features)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Imbalance features
    # -------------------------------------------------------------------------------------------------------------------------------------------

    imbalance_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()
        # Volume imbalance ratio to see if there is more buying pressure
        imbalance_features[f"quote_imbalance_ratio_{offset}h"] = df_window["quote_sign"].sum() / df_window["quote_abs"].sum()
        # Imbalance ratio in slippages to see if there is skew towards long slippages
        imbalance_features[f"quote_slippage_imbalance_ratio_{offset}h"] = df_window["quote_slippage_sign"].sum() / df_window["quote_slippage_abs"].sum()

    all_features.update(imbalance_features)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Imbalance features
    # -------------------------------------------------------------------------------------------------------------------------------------------

    evt_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()

        evt_features[f"quote_abs_powerlaw_alpha_{offset}h"] = estimate_alpha_mle(
            data=df_window["quote_abs"]
        )

    all_features.update(evt_features)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Liquidity features
    # -------------------------------------------------------------------------------------------------------------------------------------------

    liquidity_features: Dict[str, float] = {}

    df_slip: pd.DataFrame = df_trades[df_trades["quote_slippage_abs"] > 0].copy()

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_slip[df_slip["time"] >= time_ub - timedelta(hours=offset)].copy()
        liquidity_features[f"linear_liquidity_slope_{offset}h"] = calculate_linear_liquidity_index(df_trades=df_window)

    all_features.update(liquidity_features)


    daily_offsets: List[int] = [1, 3, 7, 14, 30]

    liquidity_proba_features: Dict[str, float] = {}

    for offset in daily_offsets:
        df_window: pd.DataFrame = df_trades[df_trades["time"] >= time_ub - timedelta(days=offset)].copy()
        a, b = regress_proba_slippage_on_quote_abs(df_trades=df_window)

        liquidity_proba_features[f"has_slippage_slope_{offset}d"] = b
        liquidity_proba_features[f"has_slippage_intercept_{offset}d"] = a

    all_features.update(liquidity_proba_features)

    # Output features
    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features


