from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base
from sqlalchemy import Float, DateTime, String, ForeignKey
from datetime import datetime

import pytz

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    trade_id: Mapped[int] = mapped_column()
    price: Mapped[float] = mapped_column(Float(precision=10))
    qty: Mapped[float] = mapped_column(Float(precision=10))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=pytz.utc))
    isBuyerMaker: Mapped[bool] = mapped_column()
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id"))

    # ticker = relationship("Ticker", back_populates="trades")


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(15), unique=True)

    # trades = relationship("Trade", back_populates="ticker")
