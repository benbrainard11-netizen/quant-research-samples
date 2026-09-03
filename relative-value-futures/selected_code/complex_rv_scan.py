"""Where does the structural (cointegration) RV edge live, by complex?

Reuses the validated cointegration machinery. For every WITHIN-complex pair
(energy, rates curve, metals, grains, FX, equity), computes in-sample ADF +
half-life + OOS Sharpe, then builds a diversified market-neutral book of all
within-complex cointegrated pairs and measures it OOS.

Run: backend/.venv/Scripts/python.exe experiments/xsectional_rv_v0/complex_rv_scan.py
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from cointegration_select import R, LOGP, SPLIT, EG_CRIT, net_series, sharpe, adf_halflife

COMPLEXES = {
    "energy": ["CL", "BZ", "HO", "RB", "NG"],
    "rates": ["ZT", "ZF", "ZN", "ZB"],
    "metals": ["GC", "SI", "HG"],
    "grains": ["ZC", "ZS", "ZW"],
    "fx": ["6E", "6B", "6J", "6A", "6C", "6S", "6N"],
    "equity": ["ES", "NQ", "YM", "RTY"],
}


def is_adf(a: str, b: str):
    A = LOGP[a][LOGP.index < SPLIT]
    B = LOGP[b][LOGP.index < SPLIT]
    beta = np.cov(A, B)[0, 1] / np.var(B)
    return adf_halflife(A - beta * B)


def main() -> int:
    coint_nets = []
    print(f"{'pair':12} {'ADF':>7} {'half-life':>9} {'OOS_Sh':>7}   (* = cointegrated, ADF<{EG_CRIT})")
    for cx, members in COMPLEXES.items():
        syms = [f"{m}.c.0" for m in members if f"{m}.c.0" in R.columns]
        rows, cx_coint = [], []
        for a, b in itertools.combinations(syms, 2):
            adf, hl = is_adf(a, b)
            net = net_series(a, b)
            oos = sharpe(net[net.index >= SPLIT])
            rows.append((a, b, adf, hl, oos))
            if adf < EG_CRIT:
                cx_coint.append(net)
                coint_nets.append(net)
        print(f"\n-- {cx.upper()} --")
        for a, b, adf, hl, oos in sorted(rows, key=lambda r: r[2]):
            star = " *" if adf < EG_CRIT else "  "
            print(f"  {a[:-4]+'/'+b[:-4]:10}{star} {adf:>7.2f} {hl:>9.1f} {oos:>7.2f}")
        if cx_coint:
            book = pd.concat(cx_coint, axis=1).mean(axis=1)
            print(f"  -> {cx} cointegrated book ({len(cx_coint)} pairs): OOS Sharpe {sharpe(book[book.index >= SPLIT]):+.2f}")

    if coint_nets:
        book = pd.concat(coint_nets, axis=1).mean(axis=1)
        oos = book[book.index >= SPLIT]
        eq = oos.cumsum()
        dd = (eq - eq.cummax()).min()
        print(f"\n=== DIVERSIFIED within-complex cointegrated book ({len(coint_nets)} pairs) ===")
        print(f"   OOS Sharpe {sharpe(oos):+.2f}   OOS ann.ret {oos.mean()*252:+.3%}   OOS maxDD {dd:.3f}")
        print(f"   full-sample Sharpe {sharpe(book):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
