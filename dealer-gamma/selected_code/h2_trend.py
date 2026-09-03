"""H2 - short-gamma TREND leg: buy a 0DTE option on the zero-gamma flip break.

Entry (frozen, PROGRAM.md): eligible when the prior-day regime is short-gamma.
Trigger = first cross of `zero_gamma` by the SPX proxy after 10:00 ET; buy the
~30-delta 0DTE option in the break direction (long call on an up-cross, long put
on a down-cross), filled at ask. Exit (first to occur): same-side wall touched,
flip re-cross, premium <= 50% of cost, 15:30 ET time stop, else settle at close.
Risk is the premium paid, so r_multiple = pnl_pts / entry_cost.
"""
import pandas as pd

from . import config, panel_io
from .regime import TREND


def _first_cross(path, arm_ms, zg):
    """First minute after arm the proxy crosses zg. Returns (ms, 'up'|'down')."""
    fwd = path[path["ms"] >= arm_ms].reset_index(drop=True)
    if fwd.empty:
        return None
    start_above = float(fwd["spx"].iloc[0]) >= zg
    for _, r in fwd.iloc[1:].iterrows():
        if start_above and r["spx_lo"] <= zg:
            return int(r["ms"]), "down"
        if not start_above and r["spx_hi"] >= zg:
            return int(r["ms"]), "up"
    return None


def _opt_bid_path(df0, strike, right, from_ms):
    """ms -> bid for the chosen contract from `from_ms` on (live quotes only)."""
    r = df0[(df0["ms_of_day"] >= from_ms) & (df0["strike"] == strike) &
            (df0["right"] == right) & (df0["bid"] > 0)]
    return dict(zip(r["ms_of_day"].astype(int), r["bid"].astype(float)))


def run_day(date_int, reg, path, df0, flip, *, require_trend=True):
    """One directional trade for one day. Result dict or None (no trade).

    `flip` = the real intraday net-GEX flip level (gamma_flip.daily_flip),
    computed once per day at the arm minute. `require_trend=False` ignores the
    regime gate — used by the matched control."""
    if require_trend and reg["regime"] != TREND:
        return None
    if flip is None or pd.isna(flip):
        return None
    cross = _first_cross(path, config.H2_ARM_MS, float(flip))
    if cross is None:
        return None
    cms, direction = cross
    right = "C" if direction == "up" else "P"
    target = reg["call_wall"] if direction == "up" else reg["put_wall"]
    s_cross = float(path[path["ms"] == cms]["spx"].iloc[0])
    snap = panel_io.snapshot(df0, cms)
    pick = panel_io.pick_by_delta(snap, right, s_cross, config.H2_LONG_DELTA, cms)
    if pick is None:
        return None
    strike, entry_cost = float(pick["strike"]), float(pick["ask"])   # buy at ask
    if entry_cost <= 0:
        return None
    return _resolve(date_int, reg, path, df0, cms, direction, right, strike,
                    target, entry_cost, float(flip), s_cross)


def _resolve(date_int, reg, path, df0, cms, direction, right, strike, target,
             entry_cost, zg, s_cross):
    bids = _opt_bid_path(df0, strike, right, cms)
    fwd = path[path["ms"] > cms]
    exit_px, exit_kind = None, "settle"
    for _, r in fwd.iterrows():
        ms = int(r["ms"])
        bid = bids.get(ms)
        # premium stop and time stop checked before target (house rule 8: stop
        # wins on ambiguity). No flip-recross stop: it sits at the entry level
        # and strangled the trend before it could develop (verified 2026-07-20).
        if bid is not None and bid <= config.H2_STOP_PREMIUM_FRAC * entry_cost:
            exit_px, exit_kind = (bid, "prem_stop"); break
        if ms >= config.H2_TIME_STOP_MS:
            exit_px, exit_kind = (bid, "time_stop"); break
        hit = (r["spx_hi"] >= target) if direction == "up" else (r["spx_lo"] <= target)
        if hit:
            exit_px, exit_kind = (bid, "target"); break
    if exit_kind == "settle" or exit_px is None:
        s_close = float(path["spx"].iloc[-1])
        intrinsic = (max(s_close - strike, 0.0) if right == "C"
                     else max(strike - s_close, 0.0))
        exit_px = intrinsic if exit_px is None else exit_px
        if exit_kind != "settle":            # stop fired but no live bid -> intrinsic
            s_at = float(fwd[fwd["ms"] >= cms]["spx"].iloc[0])
            exit_px = (max(s_at - strike, 0.0) if right == "C"
                       else max(strike - s_at, 0.0))
    legs = 1 if exit_kind == "settle" else 2
    pnl_pts = exit_px - entry_cost
    pnl_usd = pnl_pts * config.SPX_MULTIPLIER - legs * config.COMMISSION_PER_LEG
    return {"date": date_int, "leg": "H2_trend", "regime": reg["regime"],
            "gex": reg["gex"], "direction": direction, "strike": strike,
            "entry_cost_pts": entry_cost, "exit": exit_kind,
            "pnl_pts": pnl_pts, "pnl_usd": pnl_usd,
            "r_multiple": pnl_pts / entry_cost}
