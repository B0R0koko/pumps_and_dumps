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

    df_ticker["quote"] = df_ticker["qty"] * df_ticker["price"]
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

    # calculate slippages
    df_trades["quote_slippage_abs"] = (
        df_trades["quote_sign"] - df_trades["qty_sign"] * df_trades["price_first"]
    )
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])

    del df_ticker
    gc.collect()

    df_trades["is_long"] = df_trades["qty_sign"] >= 0
    df_trades["quote_long"] = df_trades["quote_abs"] * df_trades["is_long"]
    df_trades["quote_short"] = df_trades["quote_abs"] * ~df_trades["is_long"]

    df_trades = df_trades.reset_index()

    # define upper bound up to which our model has data
    time_ub: pd.Timestamp = pump.time - timedelta(minutes=15)

    hour_offsets: List[int] = [1, 3, 6, 12, 24, 48, 60, 72]

    df_hourly_candles: pd.DataFrame = df_trades.resample(
        on="time", rule="1h", closed="left"
    ).agg(
        open=("price_first", "first"),
        close=("price_last", "last"),
        low=("price_min", "min"),
        high=("price_max", "max"),
        volume=("qty_abs", "sum"),
        volume_quote=("quote_abs", "sum"),
        quote_slippage_abs=("quote_slippage_abs", "sum"),
    ).reset_index()

    df_hourly_candles["log_return"] = np.log(df_hourly_candles["close"] / df_hourly_candles["close"].shift(1))

    # find long_run std values
    long_run_volume_std: float = df_hourly_candles["volume"].std()
    long_run_volume_quote_std: float = df_hourly_candles["volume_quote"].std()

    hourly_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window: pd.DataFrame = df_hourly_candles[df_hourly_candles["time"] >= time_ub - timedelta(hours=offset)].copy()
        # calculate features over these windows
        hourly_features[f"overall_return_{offset}h"] = (df_window["log_return"] + 1).prod() # overall log return
        
        if offset == 1:
            continue

        hourly_features[f"log_return_std_{offset}h"] = df_window["log_return"].std()
        hourly_features[f"volume_std_{offset}h_SLR"] = df_window["volume"].std() / long_run_volume_std # see how volatility in volume relates to long run volatility
        hourly_features[f"volume_quote_std_{offset}h_SLR"] = df_window["volume_quote"].std() / long_run_volume_quote_std

    # indicator variable of the last hour being highest in volume
    df_hourly_candles_1d: pd.DataFrame = df_hourly_candles[
        df_hourly_candles["time"] >= time_ub - timedelta(days=1)
    ].copy()

    hourly_features["is_last_hour_highest_quote_vol_24h"] = (
        (df_hourly_candles_1d["volume_quote"].iloc[-1] == df_hourly_candles_1d["volume_quote"].max()).astype(int)
    )

    all_features.update(hourly_features)

    # find quote slippage for each trade
    df_trades["quote_slippage_abs"] = (
        df_trades["quote_sign"] - df_trades["qty_sign"] * df_trades["price_first"]
    )
    df_trades["quote_slippage_sign"] = df_trades["quote_slippage_abs"] * np.sign(df_trades["qty_sign"])

    df_trades["quote_slippage_long"] = df_trades["quote_slippage_abs"] * df_trades["is_long"]
    df_trades["quote_slippage_short"] = df_trades["quote_slippage_abs"] * (~df_trades["is_long"])

    df_72h = df_trades[
        df_trades["time"] >= time_ub - timedelta(hours=72)
    ].copy()

    quote_slippage_features: Dict[str, float] = {}
    powerlaw_features: Dict[str, float] = {}

    cutoff_percentile: float = 0.99 # fit powerlaw to higher quantiles
    
    for offset in hour_offsets:
        df_window = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()

        quote_slippage_features[f"quote_slippage_imbalance_{offset}h"] = df_window["quote_slippage_sign"].sum() / df_window["quote_slippage_abs"].sum()
        # Share of slippage of this window in total 72h window 
        quote_slippage_features[f"quote_slippage_share_{offset}h_72h"] = df_window["quote_slippage_abs"].sum() / df_72h["quote_slippage_abs"].sum()
        # Share of long slippages in total 72h window, we expect higher values for each window for pumped tickers
        quote_slippage_features[f"quote_slippage_long_share_{offset}h_72h"] = (
            df_window["quote_slippage_long"].sum() / df_72h["quote_slippage_long"].sum()
        )
        # Fit powerlaw to measure spread in the upper quantiles
        quote_slippage_long: pd.Series = df_window[
            df_window["quote_slippage_long"] >= df_window["quote_slippage_long"].quantile(cutoff_percentile)
        ]["quote_slippage_long"]

        powerlaw_features[f"quote_slippage_long_powerlaw_alpha_{offset}h"] = (
            powerlaw.fit(quote_slippage_long)[0] + 1
        )

    # check if the last hour prior to the pump is highest in the last 24hours in terms quote slippages
    quote_slippage_features["is_last_hour_highest_quote_slippage_abs"] = (
        (df_hourly_candles_1d["quote_slippage_abs"].iloc[-1] == df_hourly_candles_1d["quote_slippage_abs"].max()).astype(int)
    )

    all_features.update(quote_slippage_features)
    all_features.update(powerlaw_features)


    volume_imbalance_features: Dict[str, float] = {}

    for offset in hour_offsets:
        df_window = df_trades[df_trades["time"] >= time_ub - timedelta(hours=offset)].copy()
        volume_imbalance_features[f"volume_quote_imbalance_{offset}h"] = df_window["quote_sign"].sum() / df_window["quote_abs"].sum()
    
    all_features.update(volume_imbalance_features)

    short_run_features: Dict[str, float] = {}

    df_1h = df_trades[df_trades["time"] >= time_ub - timedelta(hours=1)].copy()
    df_24h = df_trades[df_trades["time"] >= time_ub - timedelta(hours=24)].copy()

    short_run_features["quote_slippage_long_share_1h_24h"] = df_1h["quote_slippage_long"].sum() / df_24h["quote_slippage_long"].sum()
    # Share of trades in the last hour in the last 24 hours
    short_run_features["num_trades_share_1h_24h"] = df_1h.shape[0] / df_24h.shape[0]
    short_run_features["num_long_trades_share_1h_24h"] = df_1h["is_long"].sum() / df_24h["is_long"].sum()

    all_features.update(short_run_features)

    df_features = pd.DataFrame(data=all_features.values()).T
    df_features.columns = all_features.keys()

    df_features["num_prev_pumps"] = check_num_times_has_been_pumped(pump=pump, ticker=ticker)

    return df_features

