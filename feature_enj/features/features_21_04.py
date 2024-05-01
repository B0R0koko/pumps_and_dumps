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

import warnings

warnings.filterwarnings("ignore")


def transform_to_features_21(
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

    df_trades_1d: pd.DataFrame = df_trades[
        df_trades["time"] >= pump.time.floor("1h") - timedelta(days=1)
    ].copy()

    for i in range(1, 5):
        df_interval: pd.DataFrame = df_trades_1d[
            df_trades_1d["time"] >= pump.time.floor("1h") - timedelta(minutes=15*i)
        ].copy()

        features[f"share_num_long_trades_{15*i}min_1d"] = df_interval["is_long"].sum() / df_trades_1d["is_long"].sum()
        features[f"share_trading_vol_{15*i}min_1d"] = df_interval["quote_abs"].sum() / df_trades_1d["quote_abs"].sum()
        features[f"share_num_trades_{15*i}min_1d"] = df_interval.shape[0] / df_trades_1d.shape[0]
        features[f"imbalance_ratio_{15*i}min"] = df_interval["quote_sign"].sum() / df_interval["quote_abs"].sum()

    df_trades_1d["rolling_999_quantile"] = df_trades_1d.rolling(on="time", window="1h")["quote_sign"].quantile(.999)
    df_trades_1d["rolling_001_quantile"] = df_trades_1d.rolling(on="time", window="1h")["quote_sign"].quantile(.001)

    features["rolling_bull_vs_bear_1d"] = np.trapz(df_trades_1d["rolling_999_quantile"]) + np.trapz(df_trades_1d["rolling_001_quantile"])

    df_features = pd.DataFrame(data=features.values()).T
    df_features.columns = features.keys()


    return df_features
