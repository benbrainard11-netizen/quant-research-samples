"""0DTE SPXW option-panel reader, BS-delta strike selection, and honest fills.

Corruption-robust: `good_dates` skips partitions whose parquet footer is invalid
(the 2025-2026 rebuild was interrupted; see PROGRAM.md data-status).

Fill law (house rule 8 + 0dte-lab): BUY at ask, SELL at bid, always. No midpoint
fill ever reaches a P&L number; any mid is reference-only. Commission per leg.
"""
import glob
import math
import os

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from . import config

_MS_PER_YEAR = 365.0 * 24 * 3600 * 1000


def _footer_ok(f: str) -> bool:
    try:
        pq.read_metadata(f)
        return True
    except Exception:
        return False


def good_dates(year=None):
    """Panel partition dates whose files all parse; optionally filter to a year."""
    out = []
    for x in sorted(os.listdir(config.PANEL_ROOT)):
        if not x.startswith("date="):
            continue
        d = int(x.split("=")[1])
        if year is not None and d // 10000 != year:
            continue
        files = glob.glob(str(config.PANEL_ROOT / x / "*.parquet"))
        if files and all(_footer_ok(f) for f in files):
            out.append(d)
    return out


def read_full(date_int: int) -> pd.DataFrame:
    """All quote rows (every expiration) for a good partition date."""
    files = glob.glob(str(config.PANEL_ROOT / f"date={date_int}" / "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["right"] = df["right"].astype(str)
    return df


def read_0dte(date_int: int) -> pd.DataFrame:
    """All 0DTE (expiration==date) quote rows for a good partition date."""
    df = read_full(date_int)
    return df[df["expiration"] == int(date_int)].copy()


def tte_years(ms: int) -> float:
    return max(config.RTH_CLOSE_MS - ms, 0) / _MS_PER_YEAR


def bs_delta(right: str, S: float, K: float, iv: float, tte: float) -> float:
    """Black-Scholes delta (r=q=0; forward=spot via parity). NaN if unpriceable."""
    if not (S > 0 and K > 0 and iv > 0 and tte > 0):
        return float("nan")
    d1 = (math.log(S / K) + 0.5 * iv * iv * tte) / (iv * math.sqrt(tte))
    return float(norm.cdf(d1)) if right == "C" else float(norm.cdf(d1) - 1.0)


def snapshot(df0dte: pd.DataFrame, ms: int) -> pd.DataFrame:
    """Live-quote rows at one ET minute (bid>0 and ask>0)."""
    s = df0dte[df0dte["ms_of_day"] == int(ms)]
    return s[(s["bid"] > 0) & (s["ask"] > 0)].copy()


def pick_by_delta(snap: pd.DataFrame, right: str, S: float, target_delta: float,
                  ms: int):
    """Nearest listed strike to target |delta| on the given side; None if none."""
    s = snap[snap["right"] == right].copy()
    if s.empty:
        return None
    tte = tte_years(ms)
    s["delta"] = [bs_delta(right, S, k, iv, tte)
                  for k, iv in zip(s["strike"], s["implied_vol"])]
    s = s.dropna(subset=["delta"])
    if s.empty:
        return None
    tgt = target_delta if right == "C" else -abs(target_delta)
    s["dist"] = (s["delta"] - tgt).abs()
    return s.nsmallest(1, "dist").iloc[0]


def quote_at(df0dte: pd.DataFrame, ms: int, strike: float, right: str):
    """The (bid, ask) for one contract at one minute, or None if not live."""
    r = df0dte[(df0dte["ms_of_day"] == int(ms)) &
               (df0dte["strike"] == strike) &
               (df0dte["right"] == right)]
    r = r[(r["bid"] > 0) & (r["ask"] > 0)]
    if r.empty:
        return None
    return float(r["bid"].iloc[0]), float(r["ask"].iloc[0])
