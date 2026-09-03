"""ES aggressor orderflow (net-delta imbalance) from Databento MBP-1.

Sign VERIFIED 2026-07-21: trade side='B' = buy aggressor (+delta), 'A' = sell
aggressor (-delta); corr(per-minute net_delta, price change) = +0.65 on ES.
The trigger signal is a trailing-window imbalance ratio in [-1, +1]:
    ratio = (buy_vol - sell_vol) / (buy_vol + sell_vol)  over OF_WINDOW_MIN.
MBP-1 is ES-only and starts 2025-05, so this layer only covers 2025-05+.

Contemporaneous correlation is NOT predictive edge — whether the ratio predicts
FORWARD direction is exactly what H3 tests (the lab's prior says it does not).
"""
import glob

import numpy as np
import pandas as pd

from . import config


def _iso(date_int: int) -> str:
    s = str(int(date_int))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def per_minute(date_int: int):
    """RTH per-minute [ms, net_delta, volume] of ES trades, or None if missing."""
    files = glob.glob(str(config.MBP1_ES / f"date={_iso(date_int)}" / "*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0], columns=["ts_event", "action", "side", "size"])
    t = df[df["action"] == "T"]
    if t.empty:
        return None
    et = t["ts_event"].dt.tz_convert(config.TZ)
    ms = ((et.dt.hour * 3600 + et.dt.minute * 60) * 1000).astype("int64")
    signed = np.where(t["side"].to_numpy() == "B", 1,
                      np.where(t["side"].to_numpy() == "A", -1, 0)) * t["size"].to_numpy()
    g = pd.DataFrame({"ms": ms.to_numpy(), "signed": signed,
                      "size": t["size"].to_numpy()})
    g = g[(g["ms"] >= config.RTH_OPEN_MS) & (g["ms"] <= config.RTH_CLOSE_MS)]
    out = g.groupby("ms").agg(net_delta=("signed", "sum"),
                              volume=("size", "sum")).reset_index()
    return out


def imbalance_series(date_int: int, window_min=None):
    """RTH [ms, ratio] trailing net-delta imbalance in [-1,1], or None."""
    win = window_min or config.OF_WINDOW_MIN
    pm = per_minute(date_int)
    if pm is None or pm.empty:
        return None
    full = pd.DataFrame({"ms": np.arange(config.RTH_OPEN_MS, config.RTH_CLOSE_MS + 1,
                                         60000, dtype="int64")})
    m = full.merge(pm, on="ms", how="left").fillna({"net_delta": 0.0, "volume": 0.0})
    roll_nd = m["net_delta"].rolling(win, min_periods=1).sum()
    roll_v = m["volume"].rolling(win, min_periods=1).sum()
    m["ratio"] = np.where(roll_v > 0, roll_nd / roll_v, 0.0)
    return m[["ms", "ratio"]]
