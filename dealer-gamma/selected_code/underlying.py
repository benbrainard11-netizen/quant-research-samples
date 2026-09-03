"""Intraday SPX proxy from the ES front-month future, ET-aligned, leak-clean.

SPX has no local intraday cash feed (paywalled), so we reconstruct it from ES:
    basis   = ES_prevclose - SPX_prevclose      (known at T-1 close)
    SPX_t   = ES_t - basis
Both anchors are prior-session values, so the intraday path carries no lookahead
(unlike the panel's parity underlying, which uses a whole-day basis). ES bars are
UTC; we convert to America/New_York and key by ET ms-of-day.
"""
import functools
import os

import pandas as pd

from . import config


@functools.lru_cache(maxsize=1)
def _es_dates():
    return sorted(x.split("=")[1] for x in os.listdir(config.ES_BARS)
                  if x.startswith("date="))


def _iso(date_int: int) -> str:
    s = str(int(date_int))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def _read_es_rth(iso: str, cols=("close",)):
    p = config.ES_BARS / f"date={iso}"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["ts_event", *cols])
    et = df["ts_event"].dt.tz_convert(config.TZ)
    df["ms"] = ((et.dt.hour * 3600 + et.dt.minute * 60) * 1000).astype("int64")
    df = df[(df["ms"] >= config.RTH_OPEN_MS) & (df["ms"] <= config.RTH_CLOSE_MS)]
    return df[["ms", *cols]].sort_values("ms").reset_index(drop=True)


def _prev_iso(iso: str):
    ds = _es_dates()
    prior = [d for d in ds if d < iso]
    return prior[-1] if prior else None


def spx_intraday(date_int: int, spx_prevclose: float):
    """RTH DataFrame [ms, spx, spx_hi, spx_lo, es] for one trade date.

    spx_hi/spx_lo carry the ES bar's high/low through the same prior-close basis,
    for honest intraday touch detection on stops/targets. None if bars missing.
    """
    iso = _iso(date_int)
    today = _read_es_rth(iso, cols=("close", "high", "low"))
    piso = _prev_iso(iso)
    prev = _read_es_rth(piso) if piso else None
    if today is None or today.empty or prev is None or prev.empty:
        return None
    es_prevclose = float(prev["close"].iloc[-1])   # last RTH bar (<=16:00 ET) T-1
    basis = es_prevclose - float(spx_prevclose)
    out = today.copy()
    out["spx"] = out["close"] - basis
    out["spx_hi"] = out["high"] - basis
    out["spx_lo"] = out["low"] - basis
    return out.rename(columns={"close": "es"})[
        ["ms", "spx", "spx_hi", "spx_lo", "es"]]
