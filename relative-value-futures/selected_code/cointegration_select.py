"""Does STRUCTURAL (cointegration) selection systematize the pairs edge?

Sharpe-based pair selection failed OOS (corr IS/OOS Sharpe = +0.12). Cointegration is
a structural property, not a performance artifact, so it should generalize better.
For each pair, on IN-SAMPLE only (<2023): Engle-Granger ADF t-stat on the spread
(more negative = more cointegrated) + mean-reversion half-life. Select by structure,
measure OOS. If corr(ADF, OOS Sharpe) >> +0.12 and the cointegrated book holds OOS,
it's a real systematic book; if not, the edge is just the few obvious pairs.

Run: backend/.venv/Scripts/python.exe experiments/xsectional_rv_v0/cointegration_select.py
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
EG_CRIT = -3.34  # Engle-Granger 5% critical value (1 regressor, estimated beta)


def net_series(a: str, b: str) -> pd.Series:
    A, B = LOGP[a], LOGP[b]
    beta = A.rolling(BETAWIN).cov(B) / B.rolling(BETAWIN).var()
    z = ((A - beta * B) - (A - beta * B).rolling(ZWIN).mean()) / (A - beta * B).rolling(ZWIN).std()
    pos = -(z / 2.0).clip(-1.0, 1.0)
    pnl = pos.shift(1) * (R[a] - beta * R[b])
    return pnl - pos.diff().abs() * (2.0 * COST_BPS) / 1e4


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * ANN) if len(x) > 50 and x.std() > 0 else np.nan


def adf_halflife(spread: pd.Series):
    """Engle-Granger ADF t-stat (no lags) + half-life on a spread series."""
    s = spread.dropna()
    ds = s.diff().dropna()
    slag = s.shift(1).loc[ds.index]
    X = np.column_stack([np.ones(len(ds)), slag.to_numpy()])
    y = ds.to_numpy()
    try:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        s2 = float(resid @ resid) / (len(ds) - 2)
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    except Exception:
        return np.nan, np.inf
    t = b[1] / se if se > 0 else np.nan
    hl = float(-np.log(2) / b[1]) if b[1] < 0 else np.inf
    return float(t), hl


def main() -> int:
    pairs = list(itertools.combinations(R.columns, 2))
    nets, rows = {}, []
    for a, b in pairs:
        A_is = LOGP[a][LOGP.index < SPLIT]
        B_is = LOGP[b][LOGP.index < SPLIT]
        beta_is = np.cov(A_is, B_is)[0, 1] / np.var(B_is)
        t, hl = adf_halflife(A_is - beta_is * B_is)
        n = net_series(a, b)
        nets[(a, b)] = n
        rows.append({"a": a[:-4], "b": b[:-4], "pair": (a, b), "adf": t, "hl": hl,
                     "is_sh": sharpe(n[n.index < SPLIT]), "oos_sh": sharpe(n[n.index >= SPLIT])})
    df = pd.DataFrame(rows).dropna(subset=["adf", "oos_sh"])

    print(f"pairs with valid ADF: {len(df)}")
    print(f"\n** corr(ADF t-stat, OOS Sharpe)   = {df['adf'].corr(df['oos_sh']):+.3f}  (more-negative ADF should -> higher OOS, so expect NEGATIVE)")
    print(f"   corr(in-sample Sh, OOS Sharpe) = {df['is_sh'].corr(df['oos_sh']):+.3f}  (the failed Sharpe-selection baseline)")

    def book_oos(sub):
        if not len(sub):
            return np.nan, 0
        bk = pd.concat([nets[p] for p in sub["pair"]], axis=1).mean(axis=1)
        return sharpe(bk[bk.index >= SPLIT]), len(sub)

    coint = df[df["adf"] < EG_CRIT]
    coint_hl = coint[(coint["hl"] > 5) & (coint["hl"] < 60)]
    print(f"\n  STRUCTURAL selection (in-sample, no lookahead):")
    sh, k = book_oos(coint); print(f"   ADF<{EG_CRIT} cointegrated ({k} pairs):          book OOS Sharpe = {sh:+.2f}")
    sh, k = book_oos(coint_hl); print(f"   + half-life 5-60d ({k} pairs):              book OOS Sharpe = {sh:+.2f}")
    for K in (10, 20):
        sh, k = book_oos(df.nsmallest(K, "adf")); print(f"   top-{K} most-cointegrated (by ADF):          book OOS Sharpe = {sh:+.2f}")

    print(f"\n  top 12 most-cointegrated pairs (in-sample ADF) -> OOS:")
    print(f"  {'pair':14} {'ADF':>7} {'half-life':>9} {'OOS_Sh':>7}")
    for _, r in df.sort_values("adf").head(12).iterrows():
        print(f"  {r['a']+'/'+r['b']:14} {r['adf']:>7.2f} {r['hl']:>9.1f} {r['oos_sh']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
