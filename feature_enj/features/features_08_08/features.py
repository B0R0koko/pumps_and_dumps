from typing import *

from datetime import datetime, timedelta
from pipes.dataloader import PumpEvent
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


import pandas as pd
import numpy as np


def exception_handler(func: Callable) -> float | None:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper

# -------------------------------------------------------------------------------------------------------------------------------------------
# Hourly features. Log returns. Volumes
# -------------------------------------------------------------------------------------------------------------------------------------------

@exception_handler
def calc_overall_return(df_window: pd.DataFrame) -> float:
    return (df_window["log_return"] + 1).prod()

@exception_handler
def calc_volume_quote_abs_zscore(df_window: pd.DataFrame, df_hourly_lr: pd.DataFrame) -> float:
    return (
        (df_window["volume_quote_abs"].mean() - df_hourly_lr["volume_quote_abs"].mean()) / df_hourly_lr["volume_quote_abs"].std()
    )

@exception_handler
def calc_volume_quote_long_share(df_window: pd.DataFrame) -> float:
    return df_window["volume_quote_abs_long"].sum() / df_window["volume_quote_abs"].sum()

@exception_handler
def calc_scaled_log_return_std(df_window: pd.DataFrame, df_hourly_lr: pd.DataFrame) -> float:
    return np.log(df_window["log_return"].std() / df_hourly_lr["log_return"].std())

@exception_handler
def calc_log_return_zscore(df_window: pd.DataFrame, df_hourly_lr: pd.DataFrame) -> float:
    return df_window["log_return"].mean() / df_hourly_lr["log_return"].std() 

# -------------------------------------------------------------------------------------------------------------------------------------------
# Slippage features
# -------------------------------------------------------------------------------------------------------------------------------------------

@exception_handler
def calc_quote_slippage_abs_share(df_window: pd.DataFrame, df_hourly_candles_120h: pd.DataFrame) -> float:
    return (
        df_window["quote_slippage_abs"].sum() / df_hourly_candles_120h["quote_slippage_abs"].sum()
    )

# -------------------------------------------------------------------------------------------------------------------------------------------
# Imbalance features
# -------------------------------------------------------------------------------------------------------------------------------------------

@exception_handler
def calc_quote_imbalance_ratio(df_window: pd.DataFrame) -> float:
    return df_window["quote_sign"].sum() / df_window["quote_abs"].sum()

@exception_handler
def calc_quote_slippage_imbalance_ratio(df_window: pd.DataFrame) -> float:
    return df_window["quote_slippage_sign"].sum() / df_window["quote_slippage_abs"].sum()

# -------------------------------------------------------------------------------------------------------------------------------------------
# Imbalance features
# -------------------------------------------------------------------------------------------------------------------------------------------

@exception_handler
def calc_quote_abs_powerlaw_alpha(df_window: pd.DataFrame) -> float:
    """Hills estimator of powerlaw alpha"""
    n = len(df_window["quote_abs"])
    xmin = min(df_window["quote_abs"])
    alpha = 1 + n / np.sum(np.log(df_window["quote_abs"] / xmin))
    return alpha

# -------------------------------------------------------------------------------------------------------------------------------------------
# Liquidity features
# -------------------------------------------------------------------------------------------------------------------------------------------

@exception_handler
def calc_linear_liquidity_slope(df_window: pd.DataFrame) -> float:
    X = df_window["quote_abs"]
    X = add_constant(X)

    Y = df_window["quote_slippage_abs"]

    model = OLS(exog=X, endog=Y).fit()
    _, b = model.params
    return b


@exception_handler
def regress_proba_slippage_on_quote_abs(df_window: pd.DataFrame) -> Tuple[float, float]:
    """Compute the effect of quote_abs size on probability that the final trade will be executed with some slippage"""
    bins = np.arange(0, df_window["quote_abs"].max(), 0.001)
    
    df_window["quote_abs_bin"] = pd.cut(df_window["quote_abs"], bins=bins)
    df_window["quote_abs_bin"] = df_window["quote_abs_bin"].apply(lambda x: x.right)
    df_window["has_slippage"] = df_window["quote_slippage_abs"] > 0 

    # Compute probas for all bins
    probas: List[dict] = []

    for bin, df_bin in df_window.groupby("quote_abs_bin", observed=True):

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


@exception_handler
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
