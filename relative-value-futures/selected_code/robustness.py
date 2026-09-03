"""Anti-overfit robustness sweep for the energy RV edge.

The edge must not depend on a lucky (BETAWIN, ZWIN, cost) pick. Sweeps the hedge-ratio window,
z window, and cost across a grid; reports OOS Sharpe for CL/BZ and the full 5-pair book (continuous,
constant-exposure book = the pure signal, no vol-target distortion). If the edge is real it should be
positive and stable across the whole grid, not a single bright cell.

Run: backend/.venv/Scripts/python.exe experiments/energy_rv_v0/robustness.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

R = pd.read_parquet("experiments/sync_regime_v0/out/daily_returns.parquet")
R.index = pd.to_datetime(R.index)
LOGP = R.cumsum()
ANN = np.sqrt(252.0)
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
LEGS = ["CL.c.0", "BZ.c.0", "HO.c.0", "RB.c.0"]
PAIRS = [p for p in itertools.combinations(LEGS, 2) if set(p) != {"HO.c.0", "RB.c.0"}]


def pair_net(a, b, betawin, zwin, cost_bps):
    beta = LOGP[a].rolling(betawin).cov(LOGP[b]) / LOGP[b].rolling(betawin).var()
    spread = LOGP[a] - beta * LOGP[b]
    z = (spread - spread.rolling(zwin).mean()) / spread.rolling(zwin).std()
    pos = -(z / 2.0).clip(-1.0, 1.0)
    return pos.shift(1) * (R[a] - beta * R[b]) - pos.diff().abs() * (2.0 * cost_bps) / 1e4


def oos_sh(x):
    x = x[x.index >= SPLIT].dropna()
    return np.nan if len(x) < 50 or x.std() == 0 else float(x.mean() / x.std() * ANN)


def book_oos(pairs, betawin, zwin, cost_bps):
    nets = [pair_net(a, b, betawin, zwin, cost_bps) for a, b in pairs]
    return oos_sh(pd.concat(nets, axis=1).mean(axis=1))


def main() -> int:
    betawins, zwins, costs = (120, 250, 375), (30, 60, 90), (1.0, 2.0, 4.0)
    for nm, pairs in [("CL/BZ", [("CL.c.0", "BZ.c.0")]), ("FULL BOOK (5 pairs)", PAIRS)]:
        print(f"\n=== {nm}: OOS Sharpe across (BETAWIN x ZWIN), 2bp cost ===")
        print("  ZWIN ->   " + "  ".join(f"{z:>6}" for z in zwins))
        cells = []
        for bw in betawins:
            row = [book_oos(pairs, bw, zw, 2.0) for zw in zwins]
            cells += row
            print(f"  BETA {bw:>4}  " + "  ".join(f"{v:>+6.2f}" for v in row))
        arr = np.array(cells)
        print(f"  -> all {len(arr)} cells: min {np.nanmin(arr):+.2f}  median {np.nanmedian(arr):+.2f}  "
              f"max {np.nanmax(arr):+.2f}  positive {np.mean(arr>0):.0%}")
        print("  cost stress (BETAWIN=250, ZWIN=60): " +
              "  ".join(f"{c:.0f}bp:{book_oos(pairs,250,60,c):+.2f}" for c in (0, 1, 2, 4, 8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
