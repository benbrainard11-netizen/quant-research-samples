"""Run H3 (regime + orderflow + 0DTE) on the 2025 train window and print the
verdict. Orderflow needs MBP-1 (2025-05+), so this is 2025-only. 2026 sealed.

    python -m experiments.gamma_0dte_v0.run_stack [--limit N]

Embeds the orderflow forward-direction pre-check (does the trigger's sign
predict the next 30 min?) so we learn whether orderflow carries ANY edge here,
per the lab's standing orderflow-null prior.
"""
import sys

import numpy as np
import pandas as pd

from . import (config, gamma_flip, h3_stack, orderflow, panel_io, regime,
               underlying)

FWD_MIN = 30


def _load_day(d, reg_row):
    path = underlying.spx_intraday(d, reg_row["spx_prevclose"])
    if path is None or path.empty:
        return None
    full = panel_io.read_full(d)
    df0 = full[full["expiration"] == int(d)].copy()
    of = orderflow.imbalance_series(d)
    return path, df0, of


def _fwd_hit(path, of, tms, ratio):
    """Did the trigger sign predict the SPX move over the next FWD_MIN?"""
    now = path[path["ms"] == tms]
    later = path[path["ms"] >= tms + FWD_MIN * 60000]
    if now.empty or later.empty:
        return None
    move = float(later["spx"].iloc[0]) - float(now["spx"].iloc[0])
    return (np.sign(move) == np.sign(ratio)) if move != 0 else None


def run(limit=None):
    dates = panel_io.good_dates(year=2025)
    if limit:
        dates = dates[:limit]
    reg = regime.regime_for_dates(dates).set_index("date")
    rows = {"h3": [], "h3_blind": [], "h3_pin": []}
    hits = []
    for d in dates:
        if d not in reg.index:
            continue
        r = reg.loc[d]
        loaded = _load_day(d, r)
        if loaded is None:
            continue
        path, df0, of = loaded
        # primary (trend) + regime-blind control
        prim = h3_stack.run_day(d, r, path, df0, of, require_trend=True)
        blind = h3_stack.run_day(d, r, path, df0, of, require_trend=False)
        if prim is not None:
            rows["h3"].append(prim)
        if blind is not None:
            rows["h3_blind"].append(blind)
            if r["regime"] == regime.PIN:
                rows["h3_pin"].append(blind)
        # orderflow forward-direction pre-check (all days with a trigger)
        if blind is not None and of is not None:
            fwd = of[of["ms"] >= config.OF_ARM_MS]
            trig = fwd[fwd["ratio"].abs() >= config.OF_THRESHOLD]
            if not trig.empty:
                h = _fwd_hit(path, of, int(trig["ms"].iloc[0]),
                             float(trig["ratio"].iloc[0]))
                if h is not None:
                    hits.append(h)
    return {k: pd.DataFrame(v) for k, v in rows.items()}, hits


def _t_zero(x):
    x = np.asarray(x, float)
    return (x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            if len(x) > 1 and x.std(ddof=1) > 0 else float("nan"))


def _welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    d = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / d if d else float("nan")


def report(res, hits):
    h3, blind, pin = res["h3"], res["h3_blind"], res["h3_pin"]
    L = ["=" * 66, "gamma_0dte_v0 H3 full stack — 2025 TRAIN (2026 sealed)", "=" * 66]
    if hits:
        hr = np.mean(hits)
        L.append(f"\nOrderflow pre-check: trigger sign predicts next {FWD_MIN}min "
                 f"direction {hr*100:.1f}% (n={len(hits)}); 50% = no edge")
    if h3.empty:
        L.append("\nH3: NO TRADES"); return "\n".join(L)
    r = h3["r_multiple"].to_numpy()
    L.append(f"\n### H3 (trend + orderflow + 0DTE)")
    L.append(f"  n={len(r)}  mean_r={r.mean():+.3f}  mean_$={h3['pnl_usd'].mean():+.1f}"
             f"  win%={(r > 0).mean()*100:.0f}  t_vs_zero={_t_zero(r):.2f}")
    if not blind.empty:
        br = blind["r_multiple"].to_numpy()
        L.append(f"  regime-blind control n={len(br)}  mean_r={br.mean():+.3f}"
                 f"  t(primary vs blind)={_welch(r, br):.2f}")
    if not pin.empty:
        pr = pin["r_multiple"].to_numpy()
        L.append(f"  pin-day control n={len(pr)}  mean_r={pr.mean():+.3f}"
                 f"  t(primary vs pin)={_welch(r, pr):.2f}")
    q = h3.assign(q=(h3["date"] % 10000 // 100 - 1) // 3 + 1).groupby("q")[
        "r_multiple"].mean()
    L.append(f"  by_quarter_r={{{', '.join(f'{int(k)}: {v:+.3f}' for k, v in q.items())}}}")
    pos = r.mean() > 0 and _t_zero(r) > 2
    beats = (not blind.empty and _welch(r, blind['r_multiple'].to_numpy()) > 2)
    L.append(f"  >>> {'PASS floors' if (pos and beats) else 'FAIL'}")
    L.append("\n" + "=" * 66)
    return "\n".join(L)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    res, hits = run(limit)
    config.OUT.mkdir(exist_ok=True)
    for k, df in res.items():
        if not df.empty:
            df.to_parquet(config.OUT / f"{k}_2025.parquet", index=False)
    print(report(res, hits))


if __name__ == "__main__":
    main()
