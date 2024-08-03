from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import timedelta
from tqdm import tqdm

from typing import *


import numpy as np
import pandas as pd
import os



@dataclass
class Ticker:
    ticker: str
    time: str

    def __post_init__(self):
        self.time: pd.Timestamp = pd.Timestamp(self.time)

    def __str__(self):
        return f"Ticker: {self.ticker} - {str(self.time)}"
    
@dataclass
class PumpEvent(Ticker):
    ticker: str
    time: str


@dataclass
class Portfolio:

    tickers: List[str]
    weights: np.ndarray[float]

    def __repr__(self) -> str:
        return f"Portfolio: {dict(zip(self.tickers, self.weights.round(4)))}"
    
    def get_weight(self, ticker: str) -> float:
        """Return weight of the asset in the portfolio"""
        return self.weights[self.tickers.index(ticker)]
    

@dataclass
class Transaction:
    ticker: str
    buy_price: float
    sell_price: float
    buy_ts: pd.Timestamp
    sell_ts: pd.Timestamp

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


class PortfolioTester(ABC):

    def __init__(self, model, data_dir: str):
        self.model = model
        self.log: pd.DataFrame = pd.DataFrame()
        self.data_dir: str = data_dir

    @abstractmethod
    def create_portfolio(self, df_probas: pd.DataFrame, pump: PumpEvent) -> Portfolio:
        """Returns a dict of portfolio weights"""

    @abstractmethod
    def sell_portfolio(self, portfolio: Portfolio, pump: PumpEvent) -> List[Transaction]:
        """returns pd.DataFrame of transactions. In this method describe all entries and exit prices"""

    def calculate_portfolio_return(self, transactions: List[Transaction], portfolio: Portfolio) -> float:
        """
            transactions = [
                Transaction(ticker="ADABTC", buy_price=0.004, sell_price=0.005, buy_ts=buy_ts, sell_ts=sell_ts}
            ]
        """
        portfolio_return: float = 0
        transaction: Transaction

        for transaction in transactions:
            asset_return = (transaction.sell_price - transaction.buy_price) / transaction.buy_price
            portfolio_return += asset_return * portfolio.get_weight(transaction.ticker)

        return portfolio_return
    
    def load_data(self, ticker: str, pivot_ts: pd.Timestamp, lookback_delta: timedelta, forward_delta: timedelta) -> pd.DataFrame:
        """Load ticke level data from local storage"""
        start: pd.Timestamp = pivot_ts - lookback_delta
        end: pd.Timestamp = pivot_ts + forward_delta

        date_range: List[pd.Timestamp] = pd.date_range(start=start, end=end, freq="D", inclusive="both").tolist()
        df: pd.DataFrame = pd.DataFrame()

        for date in date_range:
            file_name: str = f"{ticker}-trades-{date.date()}.parquet"
            df_date: pd.DataFrame = pd.read_parquet(
                os.path.join(self.data_dir, ticker, file_name)
            )
            df = pd.concat([df, df_date])

        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
        df["quote_abs"] = df["price"] * df["qty"] # calculate quote spent

        return df
    

    def get_buy_price(self, df_ticker: pd.DataFrame, buy_ts: pd.Timestamp) -> float:
        """Get the price closest to buy_ts timestamp and return the corresponding price"""
        df_ticker["ts_delta"] = np.abs(df_ticker["time"] - buy_ts)
        # find minimum abs difference in time to get the closest price
        buy_price: float = df_ticker[df_ticker["ts_delta"] == df_ticker["ts_delta"].min()]["price"].iloc[0]
        return buy_price
    

    def get_sell_price(self, df_ticker: pd.DataFrame, pump: PumpEvent) -> Tuple[float, pd.Timestamp]:
        """
        Get the sell price closest to sell_ts which is essentially pump.time. This method is called for assets that were misclassified
        by the model which are immediately sold at pump.time
        """
        df_ticker["ts_delta"] = np.abs(df_ticker["time"] - pump.time)
        # find minimum abs difference in time to get the closest price
        sell_price: float = df_ticker[df_ticker["ts_delta"] == df_ticker["ts_delta"].min()]["price"].iloc[0]
        return sell_price, pump.time
    
    
    def log_transactions(self, portfolio: Portfolio, transactions: List[Transaction], pump: PumpEvent) -> None:
        """Save log of transactions to the df_log_transactions"""
        transaction: Transaction

        df_log: pd.DataFrame = pd.DataFrame.from_dict([transaction.to_dict() for transaction in transactions])
        df_log["weight"] = portfolio.weights
        df_log["pumped_ticker"] = pump.ticker
        df_log["pump_time"] = pump.time

        self.log = pd.concat([self.log, df_log])
        

    def evaluate_crosssection(self, df_crosssection: pd.DataFrame, reg_cols: List[str], pump: PumpEvent) -> Dict[str, Any]:
        """
        1. Make a prediction using self.model.predict_proba and then take top logits to create a portfolio
        2. Create a portfolio using self.create_portfolio method 
        3. Sell portfolio using self.sell_portfolio method
        """
        probas_pred: np.array = self.model.predict_proba(df_crosssection[reg_cols])[:, 1] # take probas of the minority class
        # Create a dataset with probas and tickers
        df_probas: pd.DataFrame = pd.DataFrame({
            "ticker": df_crosssection["ticker"],
            "proba": probas_pred
        })

        portfolio: Portfolio = self.create_portfolio(df_probas=df_probas, pump=pump)
        transactions: List[Transaction] = self.sell_portfolio(portfolio=portfolio, pump=pump)
        portfolio_return: float = self.calculate_portfolio_return(transactions=transactions, portfolio=portfolio)

        # Add all transactions to log
        self.log_transactions(portfolio=portfolio, transactions=transactions, pump=pump)

        return {
            "portfolio_return": portfolio_return,
            "portfolio_contained_pump": pump.ticker in portfolio.tickers
        }
    
    def backtest(self, df_test: pd.DataFrame, reg_cols: List[str]) -> pd.DataFrame:
        """Calculates returns for each cross-section in df_test sample and returns stats on the strategy returns"""
        
        portfolio_outputs: List[Dict[str, Any]] = []
        self.log: pd.DataFrame = pd.DataFrame() # clear log on restart

        for (pumped_ticker, pump_time), df_crosssection in tqdm(df_test.groupby(["pumped_ticker", "pump_time"])):
            pump: PumpEvent = PumpEvent(ticker=pumped_ticker, time=pump_time)
            portfolio_output: Dict[str, Any] = self.evaluate_crosssection(
                df_crosssection=df_crosssection, reg_cols=reg_cols, pump=pump
            )
            portfolio_outputs.append({
                **portfolio_output,
                "pumped_ticker": pumped_ticker,
                "pump_time": pump_time,
            })
        
        df_backtest: pd.DataFrame = pd.DataFrame(portfolio_outputs)
        df_backtest = df_backtest.sort_values(by="pump_time", ascending=True).reset_index(drop=True)
        return df_backtest