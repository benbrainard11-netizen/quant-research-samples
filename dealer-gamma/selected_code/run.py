"""Run both legs (and their matched controls) over the clean panel window and
print the pre-registered verdict. Train only; the 2026 holdout stays sealed.

    python -m experiments.gamma_0dte_v0.run [year]

Control design (PROGRAM.md): each leg's edge must beat the SAME mechanics run on
the WRONG regime (pin trade on trend days / trend trade on pin days). If the
regime carries no information, primary and control expectancies match.
"""
import sys

import numpy as np
import pandas as pd

from . import config, gamma_flip, h1_pin, h2_trend, panel_io, regime, underlying


def _load_day(date_int, reg_row):
    path = underlying.spx_intraday(date_int, reg_row["spx_prevclose"])
    if path is None or path.empty:
        return None, None, None, None
    full = panel_io.read_full(date_int)
    df0 = full[full["expiration"] == int(date_int)].copy()
    s_arm = path[path["ms"] == config.H2_ARM_MS]["spx"]
    flip = (gamma_flip.daily_flip(full, date_int, config.H2_ARM_MS, float(s_arm.iloc[0]))
            if not s_arm.empty else None)
    return path, full, df0, flip


def run_backtest(year):
    dates = panel_io.good_dates(year=year)
    reg = regime.regime_for_dates(dates).set_index("date")
    rows = {"h1": [], "h1_ctrl": [], "h2": [], "h2_ctrl": []}
    for d in dates:
        if d not in reg.index:
            continue
        r = reg.loc[d]
        path, full, df0, flip = _load_day(d, r)
        if path is None:
            continue
        for key, res in (
            ("h1", h1_pin.run_day(d, r, path, df0, require_pin=True)),
            ("h1_ctrl", h1_pin.run_day(d, r, path, df0, require_pin=False)),
            ("h2", h2_trend.run_day(d, r, path, df0, flip, require_trend=True)),
            ("h2_ctrl", h2_trend.run_day(d, r, path, df0, flip, require_trend=False)),
        ):
            if res is not None:
                rows[key].append(res)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def _welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = np.sqrt(va + vb)
    return float("nan") if denom == 0 else (a.mean() - b.mean()) / denom


def _stats(df, ctrl, primary_regime):
    """Primary vs control on r_multiple, with the kill-criteria checks."""
    if df.empty:
        return {"n": 0, "verdict": "NO TRADES"}
    r = df["r_multiple"].to_numpy()
    prim = df[df["regime"] == primary_regime]["r_multiple"].to_numpy()
    ctrl_r = ctrl["r_multiple"].to_numpy() if not ctrl.empty else np.array([])
    # per-quarter concentration: is the edge one-quarter-only?
    q = df.assign(q=(df["date"] % 10000 // 100 - 1) // 3 + 1).groupby("q")[
        "r_multiple"].mean()
    t_self = (prim.mean() / (prim.std(ddof=1) / np.sqrt(len(prim)))
              if len(prim) > 1 and prim.std(ddof=1) > 0 else float("nan"))
    t_ctrl = _welch_t(prim, ctrl_r)
    pos_exp = prim.mean() > 0
    beats = (t_ctrl > 2) if not np.isnan(t_ctrl) else False
    one_q = (q > 0).sum() <= 1 if len(q) else True
    verdict = ("PASS floors" if (pos_exp and beats and t_self > 2 and not one_q)
               else "FAIL")
    return {"n": len(prim), "mean_r": prim.mean(), "mean_usd": df["pnl_usd"].mean(),
            "win_rate": (prim > 0).mean(), "t_vs_zero": t_self,
            "ctrl_n": len(ctrl_r), "ctrl_mean_r": (ctrl_r.mean() if len(ctrl_r) else float("nan")),
            "t_vs_ctrl": t_ctrl, "one_quarter_only": one_q,
            "by_quarter_r": {int(k): round(v, 3) for k, v in q.items()},
            "verdict": verdict}


def _report(res):
    lines = ["=" * 66, "gamma_0dte_v0 — TRAIN read (2026 sealed)", "=" * 66]
    for leg, prim_key, ctrl_key, prim_reg, ctrl_reg in (
        ("H1 PIN (sell condor, long-gamma days)", "h1", "h1_ctrl", regime.PIN, regime.TREND),
        ("H2 TREND (buy break, short-gamma days)", "h2", "h2_ctrl", regime.TREND, regime.PIN),
    ):
        # matched control = SAME mechanics on the OPPOSITE regime only
        ctrl = res[ctrl_key]
        if not ctrl.empty:
            ctrl = ctrl[ctrl["regime"] == ctrl_reg]
        s = _stats(res[prim_key], ctrl, prim_reg)
        lines.append(f"\n### {leg}")
        if s["n"] == 0:
            lines.append("  NO TRADES"); continue
        lines.append(
            f"  n={s['n']}  mean_r={s['mean_r']:+.3f}  mean_$={s['mean_usd']:+.1f}  "
            f"win%={s['win_rate']*100:.0f}  t_vs_zero={s['t_vs_zero']:.2f}")
        lines.append(
            f"  control n={s['ctrl_n']}  ctrl_mean_r={s['ctrl_mean_r']:+.3f}  "
            f"t_vs_ctrl={s['t_vs_ctrl']:.2f}")
        lines.append(f"  by_quarter_r={s['by_quarter_r']}")
        lines.append(f"  >>> {s['verdict']}")
    lines.append("\n" + "=" * 66)
    return "\n".join(lines)


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else config.TRAIN_YEARS[0]
    res = run_backtest(year)
    config.OUT.mkdir(exist_ok=True)
    for k, df in res.items():
        if not df.empty:
            df.to_parquet(config.OUT / f"{k}_{year}.parquet", index=False)
    print(_report(res))


if __name__ == "__main__":
    main()
