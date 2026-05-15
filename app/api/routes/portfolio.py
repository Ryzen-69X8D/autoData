from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import Holding, Portfolio, User, Watchlist
from app.db.schemas import (
    HoldingCreate,
    HoldingRead,
    HoldingUpdate,
    PortfolioCashUpdate,
    PortfolioRead,
    WatchlistRead,
    WatchlistUpdate,
)


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _get_or_create_portfolio(db: Session, user_id: int) -> Portfolio:
    portfolio = db.query(Portfolio).filter(Portfolio.user_id == user_id).first()
    if portfolio is not None:
        return portfolio

    portfolio = Portfolio(user_id=user_id, total_value=0, cash_balance=0)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def _recalculate_total(portfolio: Portfolio) -> None:
    holdings_value = sum(
        Decimal(holding.shares) * Decimal(holding.average_buy_price)
        for holding in portfolio.holdings
    )
    portfolio.total_value = Decimal(portfolio.cash_balance) + holdings_value


@router.get("", response_model=PortfolioRead)
def read_portfolio(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PortfolioRead:
    portfolio = _get_or_create_portfolio(db, current_user.id)
    _recalculate_total(portfolio)
    db.commit()
    db.refresh(portfolio)
    return PortfolioRead.model_validate(portfolio)


@router.put("/cash", response_model=PortfolioRead)
def update_cash_balance(
    payload: PortfolioCashUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PortfolioRead:
    portfolio = _get_or_create_portfolio(db, current_user.id)
    portfolio.cash_balance = payload.cash_balance
    _recalculate_total(portfolio)
    db.commit()
    db.refresh(portfolio)
    return PortfolioRead.model_validate(portfolio)


@router.post("/holdings", response_model=HoldingRead, status_code=status.HTTP_201_CREATED)
def upsert_holding(
    payload: HoldingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HoldingRead:
    portfolio = _get_or_create_portfolio(db, current_user.id)
    ticker = payload.ticker.upper()
    holding = (
        db.query(Holding)
        .filter(Holding.portfolio_id == portfolio.id, Holding.ticker == ticker)
        .first()
    )

    if holding is None:
        holding = Holding(
            ticker=ticker,
            shares=payload.shares,
            average_buy_price=payload.average_buy_price,
        )
        portfolio.holdings.append(holding)
    else:
        holding.shares = payload.shares
        holding.average_buy_price = payload.average_buy_price

    _recalculate_total(portfolio)
    db.commit()
    db.refresh(holding)
    return HoldingRead.model_validate(holding)


@router.put("/holdings/{holding_id}", response_model=HoldingRead)
def update_holding(
    holding_id: int,
    payload: HoldingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HoldingRead:
    portfolio = _get_or_create_portfolio(db, current_user.id)
    holding = (
        db.query(Holding)
        .filter(Holding.id == holding_id, Holding.portfolio_id == portfolio.id)
        .first()
    )
    if holding is None:
        raise HTTPException(status_code=404, detail="Holding not found")

    if payload.shares is not None:
        holding.shares = payload.shares
    if payload.average_buy_price is not None:
        holding.average_buy_price = payload.average_buy_price

    _recalculate_total(portfolio)
    db.commit()
    db.refresh(holding)
    return HoldingRead.model_validate(holding)


@router.delete("/holdings/{holding_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_holding(
    holding_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    portfolio = _get_or_create_portfolio(db, current_user.id)
    holding = (
        db.query(Holding)
        .filter(Holding.id == holding_id, Holding.portfolio_id == portfolio.id)
        .first()
    )
    if holding is None:
        raise HTTPException(status_code=404, detail="Holding not found")

    if holding in portfolio.holdings:
        portfolio.holdings.remove(holding)
    db.delete(holding)
    db.flush()
    _recalculate_total(portfolio)
    db.commit()


@router.get("/watchlist", response_model=WatchlistRead)
def read_watchlist(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WatchlistRead:
    watchlist = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).first()
    if watchlist is None:
        watchlist = Watchlist(user_id=current_user.id, ticker_symbols=[])
        db.add(watchlist)
        db.commit()
        db.refresh(watchlist)
    return WatchlistRead.model_validate(watchlist)


@router.put("/watchlist", response_model=WatchlistRead)
def update_watchlist(
    payload: WatchlistUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WatchlistRead:
    watchlist = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).first()
    if watchlist is None:
        watchlist = Watchlist(user_id=current_user.id)
        db.add(watchlist)

    watchlist.ticker_symbols = [symbol.upper() for symbol in payload.ticker_symbols]
    db.commit()
    db.refresh(watchlist)
    return WatchlistRead.model_validate(watchlist)
