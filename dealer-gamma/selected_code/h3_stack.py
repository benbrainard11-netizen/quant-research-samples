"""H3 - the full stack: regime FILTER + ES orderflow TRIGGER + 0DTE entry.

The architecture: regime says the day's character, orderflow gauges
intraday pressure, 0DTE expresses it. Frozen:
- eligible when prior-day regime is short-gamma (trend) [require_trend]
- trigger = first minute after 10:00 ET the trailing net-delta imbalance ratio
  crosses +/-OF_THRESHOLD; direction = its sign
- buy the ~30-delta 0DTE option in that direction at ask
- exit: same-side wall target, premium -50% stop, 15:30 time stop, else settle
Controls: same trade regime-blind and on pin days (does the regime FILTER help?).
"""
import pandas as pd

from . import config, panel_io
from .regime import TREND


def _opt_bid_path(df0, strike, right, from_ms):
    r = df0[(df0["ms_of_day"] >= from_ms) & (df0["strike"] == strike) &
            (df0["right"] == right) & (df0["bid"] > 0)]
    return dict(zip(r["ms_of_day"].astype(int), r["bid"].astype(float)))


def run_day(date_int, reg, path, df0, of_series, *, require_trend=True):
    """One orderflow-triggered 0DTE trade. Result dict or None.

    require_trend=False = regime-blind control (any regime)."""
    if require_trend and reg["regime"] != TREND:
        return None
    if of_series is None:
        return None
    fwd = of_series[of_series["ms"] >= config.OF_ARM_MS]
    trig = fwd[fwd["ratio"].abs() >= config.OF_THRESHOLD]
    if trig.empty:
        return None
    tms = int(trig["ms"].iloc[0])
    ratio = float(trig["ratio"].iloc[0])
    direction = "up" if ratio > 0 else "down"
    right = "C" if direction == "up" else "P"
    prow = path[path["ms"] == tms]
    if prow.empty:
        return None
    s0 = float(prow["spx"].iloc[0])
    target = reg["call_wall"] if direction == "up" else reg["put_wall"]
    snap = panel_io.snapshot(df0, tms)
    pick = panel_io.pick_by_delta(snap, right, s0, config.H2_LONG_DELTA, tms)
    if pick is None:
        return None
    strike, entry_cost = float(pick["strike"]), float(pick["ask"])
    if entry_cost <= 0:
        return None
    return _resolve(date_int, reg, path, df0, tms, direction, right, strike,
                    target, entry_cost, ratio)


def _resolve(date_int, reg, path, df0, tms, direction, right, strike, target,
             entry_cost, ratio):
    bids = _opt_bid_path(df0, strike, right, tms)
    exit_px, exit_kind = None, "settle"
    for _, r in path[path["ms"] > tms].iterrows():
        ms = int(r["ms"])
        bid = bids.get(ms)
        if bid is not None and bid <= config.H2_STOP_PREMIUM_FRAC * entry_cost:
            exit_px, exit_kind = bid, "prem_stop"; break
        if ms >= config.H2_TIME_STOP_MS:
            exit_px, exit_kind = bid, "time_stop"; break
        hit = (r["spx_hi"] >= target) if direction == "up" else (r["spx_lo"] <= target)
        if hit:
            exit_px, exit_kind = bid, "target"; break
    if exit_px is None:
        s_close = float(path["spx"].iloc[-1])
        exit_px = (max(s_close - strike, 0.0) if right == "C"
                   else max(strike - s_close, 0.0))
    legs = 1 if exit_kind == "settle" else 2
    pnl_pts = exit_px - entry_cost
    pnl_usd = pnl_pts * config.SPX_MULTIPLIER - legs * config.COMMISSION_PER_LEG
    return {"date": date_int, "leg": "H3_stack", "regime": reg["regime"],
            "gex": reg["gex"], "direction": direction, "of_ratio": ratio,
            "strike": strike, "entry_cost_pts": entry_cost, "exit": exit_kind,
            "pnl_pts": pnl_pts, "pnl_usd": pnl_usd,
            "r_multiple": pnl_pts / entry_cost}
