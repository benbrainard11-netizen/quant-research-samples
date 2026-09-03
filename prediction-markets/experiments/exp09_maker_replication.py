"""Experiment 09 - Does the maker edge replicate across markets?

exp08 found resting orders earned +0.0048 per contract on pitcher strikeouts,
positive but at t=+1.62. More of the same market would only narrow that one
estimate. Independent replication across separate markets is stronger
evidence, and it comes with a built-in control.

KXMLBGAME charges maker fees; every prop series does not. If the edge is
really the fee asymmetry rather than something generic about resting, it
should be visibly weaker on game markets.

Run:  python experiments/exp09_maker_replication.py
"""

from __future__ import annotations

import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sportmodels.kalshi import trading_fee

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "kalshi_history"
TRADES = ROOT / "data" / "trades"

# From the series catalog: props charge takers only, game markets charge both.
MAKER_PAYS = {"KXMLBGAME", "KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"}


def score(series: str) -> dict | None:
    tp = TRADES / f"trades_{series}.parquet"
    sp = HIST / f"settled_{series}.parquet"
    if not (tp.exists() and sp.exists()):
        return None

    t = pd.read_parquet(tp)
    settled = pd.read_parquet(sp)
    binary = settled[settled["result"].isin(["yes", "no"])].copy()
    binary["outcome"] = (binary["result"] == "yes").astype(int)

    d = t.merge(binary[["ticker", "outcome"]], on="ticker", how="inner")
    d = d.dropna(subset=["yes_price_dollars", "count_fp", "taker_side"])
    d = d[d["count_fp"] > 0]
    if len(d) < 1000:
        return None

    p = d["yes_price_dollars"].to_numpy(float)
    y = d["outcome"].to_numpy(float)
    size = d["count_fp"].to_numpy(float)
    bought_yes = (d["taker_side"] == "yes").to_numpy()

    taker = np.where(bought_yes, y - p, p - y)
    maker = -taker

    # Economically meaningful figure: total maker P&L over total contracts.
    # Clustered by market, weighting each market by the volume it carried, so
    # the point estimate and its t-statistic describe the same quantity.
    per = (pd.DataFrame({"m": maker * size, "s": size, "k": d["ticker"].to_numpy()})
           .groupby("k").sum())
    per["pnl"] = per["m"] / per["s"]
    w = per["s"] / per["s"].sum()
    mean = float((per["pnl"] * w).sum())
    var = float((w * (per["pnl"] - mean) ** 2).sum() * len(per) / max(len(per) - 1, 1))
    se = float(np.sqrt(var * (w**2).sum()))

    fee = trading_fee(np.where(bought_yes, p, 1.0 - p))
    maker_fee = fee if series in MAKER_PAYS else np.zeros_like(fee)

    return {
        "series": series,
        "maker_pays_fee": series in MAKER_PAYS,
        "markets": int(d.ticker.nunique()),
        "trades": len(d),
        "contracts": float(size.sum()),
        "maker_gross": mean,
        "t": mean / se if se else np.nan,
        "maker_net": float(np.average(maker - maker_fee, weights=size)),
        "taker_net": float(np.average(taker - fee, weights=size)),
    }


def main() -> None:
    files = sorted(glob.glob(str(TRADES / "trades_*.parquet")))
    series = [os.path.basename(f).replace("trades_", "").replace(".parquet", "")
              for f in files]
    print(f"Scoring {len(series)} series with trade tapes: {', '.join(series)}\n")

    rows = []
    for s in series:
        r = score(s)
        if r is None:
            print(f"  {s}: skipped (too few settled trades yet)")
            continue
        rows.append(r)

    if not rows:
        print("Nothing scorable yet - let the trade collectors finish.")
        return

    df = pd.DataFrame(rows).sort_values("maker_pays_fee")
    print("=== Maker P&L per contract, by series ===")
    print(df[["series", "maker_pays_fee", "markets", "trades", "contracts",
              "maker_gross", "t", "maker_net", "taker_net"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    props = df[~df["maker_pays_fee"]]
    games = df[df["maker_pays_fee"]]

    print("\n=== The control ===")
    if len(props):
        print(f"  maker-free series ({len(props)}): "
              f"mean gross {props['maker_gross'].mean():+.4f}, "
              f"{int((props['maker_gross'] > 0).sum())}/{len(props)} positive")
    if len(games):
        print(f"  maker-pays series ({len(games)}): "
              f"mean gross {games['maker_gross'].mean():+.4f}, "
              f"net {games['maker_net'].mean():+.4f}")
    else:
        print("  no maker-pays series scored yet - control still missing")

    print("\n=== Combined across independent markets ===")
    # Pool only the maker-free series. Combining the control into the effect
    # it exists to test would dilute the statistic with a series we expect to
    # read zero - it makes a working control look like a weaker result.
    ts = props["t"].dropna()
    if len(ts) > 1:
        stouffer = ts.sum() / np.sqrt(len(ts))
        print(f"  {len(ts)} maker-free series, individual t: "
              + ", ".join(f"{v:+.2f}" for v in ts))
        print(f"  Stouffer combined z = {stouffer:+.2f}")
        print("  (valid only because the series are separate markets; it would")
        print("   be meaningless applied to slices of one market)")
    if len(games):
        print(f"  control, held out: "
              + ", ".join(f"{s}={v:+.2f}" for s, v in zip(games['series'], games['t'])))

    print("\n  CAVEAT: the control differs from the props in more than its fee")
    print("  schedule - game markets are far more liquid and draw a different")
    print("  mix of participants. This is a suggestive contrast, not a clean")
    print("  isolation of the fee variable.")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "exp09_replication.csv", index=False)
    print("\nSaved -> results/exp09_replication.csv")


if __name__ == "__main__":
    main()
