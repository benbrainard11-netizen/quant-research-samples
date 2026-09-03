"""Decisive no-lookahead validation of the pairs-mean-reversion lead.

Scan ALL 325 pairs in the universe with the same honest spread backtest. Then the
real test: select the best pairs on IN-SAMPLE only (pre-2023) and measure that exact
selection OUT-OF-SAMPLE (2023+). If in-sample Sharpe predicts OOS Sharpe, the edge is
real and selectable without selection bias. Also reports corr(IS, OOS) across all pairs.

Run: backend/.venv/Scripts/python.exe experiments/xsectional_rv_v0/systematic_pairs.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

R = pd.read_parquet(Path(__file__).resolve().parents[1] / "sync_regime_v0" / "out" / "daily_returns.parquet")
R.index = pd.to_datetime(R.index)
LOGP = R.cumsum()
ANN = np.sqrt(252.0)
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
BETAWIN, ZWIN, COST_BPS = 250, 60, 2.0


def net_series(a: str, b: str) -> pd.Series:
    A, B = LOGP[a], LOGP[b]
    beta = A.rolling(BETAWIN).cov(B) / B.rolling(BETAWIN).var()
    spread = A - beta * B
    z = (spread - spread.rolling(ZWIN).mean()) / spread.rolling(ZWIN).std()
    pos = -(z / 2.0).clip(-1.0, 1.0)
    pnl = pos.shift(1) * (R[a] - beta * R[b])
    turn = pos.diff().abs()
    return pnl - turn * (2.0 * COST_BPS) / 1e4


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 50 and x.std() > 0 else np.nan


def main() -> int:
    pairs = list(itertools.combinations(R.columns, 2))
    print(f"scanning {len(pairs)} pairs over {R.shape[0]} days; IS<2023, OOS>=2023")
    nets, rows = {}, []
    for a, b in pairs:
        n = net_series(a, b)
        nets[(a, b)] = n
        rows.append({"a": a[:-4], "b": b[:-4], "pair": (a, b),
                     "is_sh": sharpe(n[n.index < SPLIT]), "oos_sh": sharpe(n[n.index >= SPLIT])})
    df = pd.DataFrame(rows).dropna(subset=["is_sh", "oos_sh"])

    gen = df["is_sh"].corr(df["oos_sh"])
    print(f"\n** generalization: corr(in-sample Sharpe, OOS Sharpe) across {len(df)} pairs = {gen:+.3f} **")
    print("   (>0 => picking pairs on in-sample history carries OOS; ~0 => it's noise)")

    print(f"\n  top 12 pairs by IN-SAMPLE Sharpe -> their OOS:")
    print(f"  {'pair':14} {'IS_Sh':>7} {'OOS_Sh':>7}")
    for _, r in df.sort_values("is_sh", ascending=False).head(12).iterrows():
        print(f"  {r['a']+'/'+r['b']:14} {r['is_sh']:>7.2f} {r['oos_sh']:>7.2f}")

    print(f"\n  selection test (rank by IS only, measure book OOS, equal-weight):")
    for K in (10, 20, 40):
        top = df.sort_values("is_sh", ascending=False).head(K)
        book = pd.concat([nets[p] for p in top["pair"]], axis=1).mean(axis=1)
        print(f"   top-{K:>2} by IS:  book OOS Sharpe = {sharpe(book[book.index >= SPLIT]):+.2f}")
    allbook = pd.concat(list(nets.values()), axis=1).mean(axis=1)
    print(f"   all {len(nets)} pairs: book OOS Sharpe = {sharpe(allbook[allbook.index >= SPLIT]):+.2f}")
    print(f"   pairs net-positive OOS: {int((df['oos_sh'] > 0).sum())}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
