from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import JSON, TypeDecorator
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class TickerSymbolList(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(ARRAY(String))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return []
        return list(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return list(value)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    watchlists: Mapped[list["Watchlist"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    total_value: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=Decimal("0.00"))
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=Decimal("0.00"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list["Holding"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "ticker", name="uq_portfolio_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        index=True,
    )
    ticker: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    shares: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    average_buy_price: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    ticker_symbols: Mapped[list[str]] = mapped_column(TickerSymbolList, default=list)

    user: Mapped[User] = relationship(back_populates="watchlists")
