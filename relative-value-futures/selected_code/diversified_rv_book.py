"""Does economically-grounded diversification beat the energy RV book alone?

cointegration_select.py shows pure-ADF selection drags in spurious cross-class pairs (YM/HG, ES/HG,
6B/6N all fail OOS). The honest question for a 'diversified RV bot': do the ECONOMICALLY real
structural spreads outside energy -- grain crush (corn/soy/wheat), the yield curve (ZF/ZN/ZB/ZT),
gold/silver -- actually ADD out-of-sample Sharpe on top of energy, or does energy carry everything?

Return-space only (validated daily_returns.parquet); no contract sizing -- this is the build-or-not
decision. If the diversified book clearly beats energy-only OOS, the full multi-complex bot is worth
building; if not, CL/BZ + energy IS the bot and 'diversification' is a mirage.

Run: backend/.venv/Scripts/python.exe experiments/energy_rv_v0/diversified_rv_book.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R = pd.read_parquet("experiments/sync_regime_v0/out/daily_returns.parquet")
R.index = pd.to_datetime(R.index)
LOGP = R.cumsum()
ANN = np.sqrt(252.0)
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
BETAWIN, ZWIN, COST_BPS = 250, 60, 2.0

# economically-grounded structural spreads (NOT data-mined by Sharpe/ADF)
COMPLEXES = {
    "energy": [("CL.c.0", "BZ.c.0"), ("CL.c.0", "RB.c.0"), ("CL.c.0", "HO.c.0"),
               ("BZ.c.0", "RB.c.0"), ("BZ.c.0", "HO.c.0")],          # crack + Brent-WTI (refining arb)
    "grains": [("ZC.c.0", "ZS.c.0"), ("ZS.c.0", "ZW.c.0"), ("ZC.c.0", "ZW.c.0")],  # crush / substitution
    "curve":  [("ZF.c.0", "ZN.c.0"), ("ZN.c.0", "ZB.c.0"), ("ZF.c.0", "ZT.c.0")],  # yield curve adjacents
    "metals": [("GC.c.0", "SI.c.0")],                                 # gold/silver ratio
}


def net_series(a: str, b: str) -> pd.Series:
    A, B = LOGP[a], LOGP[b]
    beta = A.rolling(BETAWIN).cov(B) / B.rolling(BETAWIN).var()
    sp = A - beta * B
    z = (sp - sp.rolling(ZWIN).mean()) / sp.rolling(ZWIN).std()
    pos = -(z / 2.0).clip(-1.0, 1.0)
    return pos.shift(1) * (R[a] - beta * R[b]) - pos.diff().abs() * (2.0 * COST_BPS) / 1e4


def stats(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    if len(x) < 50 or x.std() == 0:
        return np.nan, np.nan, np.nan
    eq = x.cumsum()
    return (x.mean() / x.std() * ANN, x.mean() * 252, (eq - eq.cummax()).min())


def book(pairs) -> pd.Series:
    cols = []
    for a, b in pairs:
        if a in R.columns and b in R.columns:
            cols.append(net_series(a, b))
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(dtype=float)


def oos(s: pd.Series) -> float:
    return stats(s[s.index >= SPLIT])[0]


def main() -> int:
    print(f"daily_returns {len(R)} days {R.index.min().date()}..{R.index.max().date()}  "
          f"OOS split {SPLIT.date()}, {COST_BPS}bp/leg\n")
    print(f"{'complex':10} {'#pairs':>6} {'full Sh':>8} {'OOS Sh':>7} {'CAGR':>7} {'maxDD':>7}")
    series = {}
    for name, pairs in COMPLEXES.items():
        s = book(pairs)
        series[name] = s
        f, c, dd = stats(s)
        o = oos(s)
        print(f"{name:10} {len(pairs):>6} {f:>8.2f} {o:>7.2f} {c:>6.1%} {dd:>7.2f}")

    # combined books
    energy_only = series["energy"]
    all_pairs = [p for ps in COMPLEXES.values() for p in ps]
    diversified = book(all_pairs)                                  # equal-weight across ALL pairs
    by_complex = pd.concat(series.values(), axis=1).mean(axis=1)   # equal-weight across COMPLEXES

    print()
    for name, s in [("ENERGY ONLY", energy_only), ("ALL pairs eq-wt", diversified),
                    ("by-COMPLEX eq-wt", by_complex)]:
        f, c, dd = stats(s)
        print(f"  {name:18} full Sh {f:+.2f} | OOS Sh {oos(s):+.2f} | CAGR {c:+.1%} | maxDD {dd:.2f}")

    e_oos, d_oos, c_oos = oos(energy_only), oos(diversified), oos(by_complex)
    best = max(d_oos, c_oos)
    print(f"\nVERDICT: energy-only OOS {e_oos:+.2f} vs diversified {best:+.2f} -> "
          + ("diversification ADDS value, build the multi-complex bot."
             if best > e_oos + 0.10 else
             "diversification does NOT beat energy meaningfully -- CL/BZ + energy IS the bot."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
