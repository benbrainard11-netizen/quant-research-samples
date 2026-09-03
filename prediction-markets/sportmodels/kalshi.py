"""Kalshi public market data.

Read-only endpoints need no credentials, which makes this the cheapest way to
start building a price history. Nothing here places orders or touches an
account - it only reads public quotes.

Prices are in cents (1-99) and represent probability directly: a 63c yes_ask
is the market asking 63% for that outcome.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Series we actually record. Explicit rather than "everything sports": the open
# universe is ~100k markets dominated by untradeable parlay stubs, and querying
# per-series turns a 25s snapshot into ~2s.
TRACKED_SERIES = (
    # MLB game-level
    "KXMLBGAME", "KXMLBTOTAL", "KXMLBF5", "KXMLBF5SPREAD", "KXMLBF7", "KXMLBEXTRAS",
    # MLB player props - the strikeout ladder is the primary research target
    "KXMLBKS", "KXMLBOUTS", "KXMLBHR", "KXMLBHRR", "KXMLBTB", "KXMLBHIT", "KXMLBRBI",
    # NFL game level, for when the season starts
    "KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL",
    # NFL player props. Unlike MLB, these have essentially no settled history
    # to backfill - 11 markets, all preseason - so this price history only
    # exists if we record it. Week 1 is the least calibrated the market will
    # be all year, and it happens once.
    "KXNFLPASSYDS", "KXNFLPASSATT", "KXNFLPASSTDS", "KXNFLPASSCOMP",
    "KXNFLPASSINT", "KXNFLRECYDS", "KXNFLRSHYDS",
)

# Kalshi series tickers are prefixed by sport. This covers the major ones.
SPORT_PREFIXES = (
    "KXNFLGAME", "KXNFL", "KXMLBGAME", "KXMLB", "KXNBAGAME", "KXNBA",
    "KXWNBAGAME", "KXWNBA", "KXNHLGAME", "KXNHL", "KXUFCFIGHT",
    "KXCFBGAME", "KXSOCCER", "KXTENNIS", "KXATP", "KXWTA",
    # Multi-venture combos: the parlay-style markets. These dominate the open
    # market count and are the ones worth pricing against their own legs.
    "KXMVESPORTS", "KXMVECROSSCATEGORY",
)


class Kalshi:
    def __init__(self, timeout: int = 30, pause: float = 0.12):
        self.session = requests.Session()
        self.timeout = timeout
        self.pause = pause  # ~8 req/s, under the read rate limit

    def _get(self, path: str, **params) -> dict:
        # Sustained bulk reads get rate limited. Without a retry those come back
        # as ordinary exceptions and the caller records a gap instead of data -
        # a ~19% silent loss when backfilling candlesticks at full speed.
        last = None
        for attempt in range(5):
            try:
                resp = self.session.get(f"{BASE}/{path}", params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2**attempt * 0.5)
                    continue
                resp.raise_for_status()
                time.sleep(self.pause)
                return resp.json()
            except Exception as e:
                last = e
                time.sleep(2**attempt * 0.5)
        raise RuntimeError(f"kalshi GET {path} failed after 5 attempts: {last}")

    def markets(self, status: str = "open", max_pages: int = 60, **params) -> pd.DataFrame:
        """Page through /markets. Returns one row per market.

        Unfiltered this walks 100,000+ markets and takes ~25s. Always pass
        `series_ticker` when you know what you want - it returns in ~0.1s.
        """
        rows, cursor = [], None
        for _ in range(max_pages):
            page = self._get("markets", limit=1000, status=status, cursor=cursor, **params)
            rows.extend(page.get("markets", []))
            cursor = page.get("cursor")
            if not cursor:
                break
        return pd.DataFrame(rows)

    def series_catalog(self, category: str = "Sports") -> pd.DataFrame:
        """All series in a category, with their fee structure."""
        return pd.DataFrame(self._get("series", category=category).get("series", []))

    def settled_markets(self, series_ticker: str, max_pages: int = 40) -> pd.DataFrame:
        """Every settled market for a series, with its `result`.

        This is retroactive history: closed markets carry their own settlement,
        so outcomes do not have to be joined from an external source.
        """
        return self.markets(status="settled", series_ticker=series_ticker, max_pages=max_pages)

    def candlesticks(self, ticker: str, start_ts: int, end_ts: int,
                     interval: int = 60) -> pd.DataFrame:
        """Historical OHLC for one market. interval is minutes (1, 60, 1440).

        Returns separate open/high/low/close for traded price, bid and ask -
        the bid/ask series is what matters, since only an executable quote can
        be acted on.
        """
        series = ticker.split("-")[0]
        j = self._get(
            f"series/{series}/markets/{ticker}/candlesticks",
            start_ts=start_ts, end_ts=end_ts, period_interval=interval,
        )
        rows = []
        for c in j.get("candlesticks", []):
            price, bid, ask = c.get("price", {}), c.get("yes_bid", {}), c.get("yes_ask", {})
            rows.append({
                "ticker": ticker,
                "end_period_ts": c.get("end_period_ts"),
                "price_open": price.get("open_dollars"),
                "price_high": price.get("high_dollars"),
                "price_low": price.get("low_dollars"),
                "price_close": price.get("close_dollars"),
                "price_mean": price.get("mean_dollars"),
                "bid_open": bid.get("open_dollars"),
                "bid_close": bid.get("close_dollars"),
                "ask_open": ask.get("open_dollars"),
                "ask_close": ask.get("close_dollars"),
                "volume": c.get("volume_fp"),
                "open_interest": c.get("open_interest_fp"),
            })
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "ticker":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Full book for one market. One call per market - use sparingly."""
        return self._get(f"markets/{ticker}/orderbook", depth=depth).get("orderbook", {})

    def trades(self, ticker: str, max_pages: int = 20) -> pd.DataFrame:
        """Full public trade tape for one market.

        `taker_side` is the key field: it says who crossed the spread, which
        means the other side of every print was someone resting an order. That
        is what makes adverse selection measurable from public data.
        """
        rows, cursor = [], None
        for _ in range(max_pages):
            page = self._get("markets/trades", ticker=ticker, limit=1000, cursor=cursor)
            batch = page.get("trades", [])
            rows.extend(batch)
            cursor = page.get("cursor")
            if not cursor or not batch:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in ("yes_price_dollars", "no_price_dollars", "count_fp"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["created_time"] = pd.to_datetime(df["created_time"], utc=True, errors="coerce")
        return df


def is_sports(ticker: str) -> bool:
    return isinstance(ticker, str) and ticker.upper().startswith(SPORT_PREFIXES)


# The API returns these as strings. Prices are already in dollars (0.00-1.00),
# so they read as probabilities directly - no cent conversion needed.
NUMERIC_FIELDS = (
    "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars",
    "last_price_dollars", "previous_price_dollars", "liquidity_dollars",
    "volume_fp", "volume_24h_fp", "open_interest_fp",
    "yes_bid_size_fp", "yes_ask_size_fp",
)


def trading_fee(price, contracts=1, rate: float = 0.07):
    """Kalshi's quadratic trading fee, in dollars.

    fee = rate * contracts * P * (1 - P), rounded up to the cent.

    Two consequences worth internalising before modelling any edge:
      - Cost peaks at 50c, where it is 1.75c per contract.
      - As a share of the price it is worst at the tails: 3.5% of a 50c
        contract but 6.3% of a 10c one. Tail rungs of a ladder are exactly
        where mispricing is most tempting and most heavily taxed.
    """
    p = np.asarray(price, dtype=float)
    return np.ceil(rate * contracts * p * (1.0 - p) * 100.0) / 100.0


def net_edge(model_prob, ask_price, contracts: int = 1):
    """Edge per contract after fees. Positive means worth a closer look."""
    p = np.asarray(model_prob, float)
    a = np.asarray(ask_price, float)
    return (p - a) * contracts - trading_fee(a, contracts)


def sports_markets(client: Kalshi | None = None, series: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Open sports markets with the quote fields worth storing.

    Queries per-series by default. Pulling the whole open universe instead
    walks 100k markets to keep a few thousand, which is why the unfiltered
    version took ~25s a snapshot.
    """
    client = client or Kalshi()
    if series is None:
        series = TRACKED_SERIES

    frames = []
    for s in series:
        try:
            part = client.markets(status="open", series_ticker=s)
        except Exception:
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    keep = [
        "ticker", "event_ticker", "title", "yes_sub_title", "status",
        "strike_type", "floor_strike", "close_time", "expiration_time",
        *NUMERIC_FIELDS,
    ]
    df = df[[c for c in keep if c in df.columns]]
    for c in NUMERIC_FIELDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # A zero on both sides means no quote, not a price of zero. Treat an
    # untradeable market as missing so it cannot masquerade as a signal.
    bid = df.get("yes_bid_dollars")
    ask = df.get("yes_ask_dollars")
    if bid is not None and ask is not None:
        no_quote = (bid <= 0) | (ask <= 0)
        df["mid_prob"] = ((bid + ask) / 2.0).mask(no_quote)
        df["spread"] = (ask - bid).mask(no_quote)
    return df.reset_index(drop=True)


def ladder_key(ticker: str) -> str:
    """Group key for a threshold ladder - the ticker minus its strike suffix.

    KXMLBKS-26JUL311915WSHATL-WSHFGRIFFIN22-7  ->  ...-WSHFGRIFFIN22
    All rows sharing this key are the same player/stat at different strikes.
    """
    return ticker.rsplit("-", 1)[0]


def check_ladder_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """Find threshold ladders whose quotes violate P(X>=k) >= P(X>=k+1).

    A ladder is a survival function, so prices must fall as the strike rises.
    Where they do not, the quotes are internally inconsistent.

    A violation is only tradeable when you can cross both sides: buying the
    lower strike at its ask and selling the higher strike at its bid locks in
    `edge` per contract, before fees. Filter hard on volume and size before
    believing any of it - most violations live in markets nobody trades.
    """
    need = {"ticker", "floor_strike", "yes_bid_dollars", "yes_ask_dollars"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing columns: {need - set(df.columns)}")

    out = []
    work = df.dropna(subset=["floor_strike"]).copy()
    work["ladder"] = work["ticker"].map(ladder_key)

    for key, grp in work.groupby("ladder"):
        grp = grp.sort_values("floor_strike")
        if len(grp) < 2:
            continue
        lo, hi = grp.iloc[:-1], grp.iloc[1:]
        # Buy the lower (more likely) strike at ask, sell the higher at bid.
        edge = hi["yes_bid_dollars"].to_numpy() - lo["yes_ask_dollars"].to_numpy()
        for i, e in enumerate(edge):
            if e > 0:
                out.append({
                    "ladder": key,
                    "buy_ticker": lo.iloc[i]["ticker"],
                    "buy_strike": lo.iloc[i]["floor_strike"],
                    "buy_ask": lo.iloc[i]["yes_ask_dollars"],
                    "sell_ticker": hi.iloc[i]["ticker"],
                    "sell_strike": hi.iloc[i]["floor_strike"],
                    "sell_bid": hi.iloc[i]["yes_bid_dollars"],
                    "edge": float(e),
                    "min_size": float(min(
                        lo.iloc[i].get("yes_ask_size_fp", 0) or 0,
                        hi.iloc[i].get("yes_bid_size_fp", 0) or 0,
                    )),
                    "volume": float(min(
                        lo.iloc[i].get("volume_fp", 0) or 0,
                        hi.iloc[i].get("volume_fp", 0) or 0,
                    )),
                })
    return pd.DataFrame(out).sort_values("edge", ascending=False) if out else pd.DataFrame()


def implied_survival(df: pd.DataFrame, price_col: str = "mid_prob") -> pd.DataFrame:
    """Reshape a ladder into P(X >= strike) per player/stat.

    This is the market's own distribution. Your model produces the same shape,
    and the comparison is distribution-to-distribution rather than
    point-estimate-to-line.
    """
    work = df.dropna(subset=["floor_strike", price_col]).copy()
    work["ladder"] = work["ticker"].map(ladder_key)
    return (
        work[["ladder", "ticker", "yes_sub_title", "floor_strike", price_col]]
        .sort_values(["ladder", "floor_strike"])
        .reset_index(drop=True)
    )
