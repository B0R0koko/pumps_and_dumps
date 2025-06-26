from typing import *

import numpy as np
import pandas as pd


def exception_handler(func: Callable) -> Optional[float]:
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
    return (df_window["volume_quote_abs"].mean() - df_hourly_lr["volume_quote_abs"].mean()) / df_hourly_lr[
        "volume_quote_abs"
    ].std()


@exception_handler
def calc_num_trades_long_share(df_window: pd.DataFrame) -> float:
    return df_window["num_trades_long"].sum() / df_window["num_trades"].sum()


@exception_handler
def calc_volume_quote_long_share(df_window: pd.DataFrame) -> float:
    return df_window["volume_quote_abs_long"].sum() / df_window["volume_quote_abs"].sum()


@exception_handler
def calc_log_return_std(df_window: pd.DataFrame, df_hourly_lr: pd.DataFrame) -> float:
    return df_window["log_return"].std() / df_hourly_lr["log_return"].std()


@exception_handler
def calc_log_return_zscore(df_window: pd.DataFrame, df_hourly_lr: pd.DataFrame) -> float:
    return df_window["log_return"].mean() / df_hourly_lr["log_return"].std()


@exception_handler
def calc_quote_slippage_abs_share(df_window: pd.DataFrame, df_hourly_candles_120h: pd.DataFrame) -> float:
    return df_window["quote_slippage_abs"].sum() / df_hourly_candles_120h["quote_slippage_abs"].sum()


@exception_handler
def calc_quote_imbalance_ratio(df_window: pd.DataFrame) -> float:
    return df_window["quote_sign"].sum() / df_window["quote_abs"].sum()


@exception_handler
def calc_quote_slippage_imbalance_ratio(df_window: pd.DataFrame) -> float:
    return df_window["quote_slippage_sign"].sum() / df_window["quote_slippage_abs"].sum()


@exception_handler
def calc_quote_abs_powerlaw_alpha(df_window: pd.DataFrame) -> float:
    """Hills estimator of powerlaw alpha"""
    n = len(df_window["quote_abs"])
    xmin = min(df_window["quote_abs"])
    alpha = 1 + n / np.sum(np.log(df_window["quote_abs"] / xmin))
    return alpha
