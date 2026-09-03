"""Pairs mean-reversion (classic stat-arb) on the obvious cointegrated futures.

Different from rank-based RV: trade the SPREAD of a specific cointegrated pair back
to its mean. Reconstruct log-price from the daily-return panel, rolling hedge-ratio
spread, continuous mean-reversion position (bet against the z-score deviation),
costs on both legs, OOS split. If even the most-arbed pairs (ES/NQ, the Treasury
curve, WTI-Brent, gold/silver) show no edge, RV is comprehensively exhausted here.

Run: backend/.venv/Scripts/python.exe experiments/xsectional_rv_v0/pairs_meanrev.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = pd.read_parquet(Path(__file__).resolve().parents[1] / "sync_regime_v0" / "out" / "daily_returns.parquet")
R.index = pd.to_datetime(R.index)
LOGP = R.cumsum()
ANN = np.sqrt(252.0)
OOS = pd.Timestamp("2023-01-01", tz="UTC")
BETAWIN, ZWIN = 250, 60

PAIRS = [
    ("ES.c.0", "NQ.c.0"), ("ES.c.0", "YM.c.0"), ("NQ.c.0", "RTY.c.0"),
    ("ZT.c.0", "ZF.c.0"), ("ZF.c.0", "ZN.c.0"), ("ZN.c.0", "ZB.c.0"),
    ("CL.c.0", "BZ.c.0"), ("HO.c.0", "RB.c.0"), ("CL.c.0", "HO.c.0"),
    ("GC.c.0", "SI.c.0"), ("6E.c.0", "6B.c.0"), ("6A.c.0", "6C.c.0"),
]


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 50 and x.std() > 0 else np.nan


def pair_pnl(a_sym: str, b_sym: str, cost_bps: float = 2.0):
    a, b = LOGP[a_sym], LOGP[b_sym]
    beta = a.rolling(BETAWIN).cov(b) / b.rolling(BETAWIN).var()
    spread = a - beta * b
    z = (spread - spread.rolling(ZWIN).mean()) / spread.rolling(ZWIN).std()
    pos = -(z / 2.0).clip(-1.0, 1.0)                 # mean-reversion: lean against the deviation
    spread_ret = R[a_sym] - beta * R[b_sym]
    pnl = pos.shift(1) * spread_ret
    turn = pos.diff().abs()
    net = pnl - turn * (2.0 * cost_bps) / 1e4        # two legs
    return pnl, net


def main() -> int:
    print(f"panel {R.shape}  pairs={len(PAIRS)}  (hedge {BETAWIN}d, z {ZWIN}d, cost 2bps/leg)")
    print(f"\n  {'pair':18} {'gross_Sh':>9} {'net_Sh':>8} {'OOS_net_Sh':>11}")
    nets = []
    for a, b in PAIRS:
        if a not in R.columns or b not in R.columns:
            continue
        g, n = pair_pnl(a, b)
        nets.append(n.rename(f"{a[:-4]}/{b[:-4]}"))
        print(f"  {a[:-4]+'/'+b[:-4]:18} {sharpe(g):>9.2f} {sharpe(n):>8.2f} {sharpe(n[n.index>=OOS]):>11.2f}")

    port = pd.concat(nets, axis=1).mean(axis=1)       # equal-weight diversified pairs book
    print(f"\n  diversified pairs book: net Sharpe full={sharpe(port):+.2f}  OOS={sharpe(port[port.index>=OOS]):+.2f}")
    print(f"  positive-net pairs: {sum(sharpe(n) > 0 for n in nets)}/{len(nets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
