"""H1 - long-gamma PIN leg: sell a defined-risk 0DTE iron condor at 10:30 ET.

Entry (frozen, PROGRAM.md): eligible when the prior-day regime is long-gamma AND
the SPX proxy at 10:30 is inside [put_wall, call_wall]. Short strikes ~15-delta
each side; long wings +/-25 pts. Honest fills: sell shorts at bid, buy wings at
ask. Exit: hold to settlement, OR full exit the first minute the proxy touches
either short strike (conservative buy-back). One trade per eligible day.

P&L is reported in SPX points and in dollars (x100 multiplier, minus commission).
"""
import pandas as pd

from . import config, panel_io
from .regime import PIN


def _quote_after(df0, ms, strike, right):
    """First live (bid, ask) at or after `ms` for a contract; None if never."""
    r = df0[(df0["ms_of_day"] >= int(ms)) & (df0["strike"] == strike) &
            (df0["right"] == right) & (df0["bid"] > 0) & (df0["ask"] > 0)]
    if r.empty:
        return None
    r = r.sort_values("ms_of_day").iloc[0]
    return float(r["bid"]), float(r["ask"])


def _first_touch(path, entry_ms, k_call, k_put):
    """First minute after entry the proxy high/low tags a short strike."""
    fwd = path[path["ms"] > entry_ms]
    hit = fwd[(fwd["spx_hi"] >= k_call) | (fwd["spx_lo"] <= k_put)]
    return int(hit["ms"].iloc[0]) if not hit.empty else None


def run_day(date_int, reg, path, df0, *, require_pin=True):
    """One condor for one day. Returns a result dict or None (no trade).

    `require_pin=False` runs the same mechanics ignoring the regime gate — used
    by the matched control (same trade on non-pin days)."""
    if require_pin and reg["regime"] != PIN:
        return None
    entry = config.H1_ENTRY_MS
    s_row = path[path["ms"] == entry]
    if s_row.empty:
        return None
    s0 = float(s_row["spx"].iloc[0])
    if not (reg["put_wall"] <= s0 <= reg["call_wall"]):
        return None
    snap = panel_io.snapshot(df0, entry)
    sc = panel_io.pick_by_delta(snap, "C", s0, config.H1_SHORT_DELTA, entry)
    sp = panel_io.pick_by_delta(snap, "P", s0, config.H1_SHORT_DELTA, entry)
    if sc is None or sp is None:
        return None
    kcs, kps = float(sc["strike"]), float(sp["strike"])
    kcl, kpl = kcs + config.H1_WING_PTS, kps - config.H1_WING_PTS
    qcl = panel_io.quote_at(df0, entry, kcl, "C")
    qpl = panel_io.quote_at(df0, entry, kpl, "P")
    if qcl is None or qpl is None or kps >= kcs:
        return None
    credit = float(sc["bid"]) + float(sp["bid"]) - qcl[1] - qpl[1]   # points in
    if credit <= 0:
        return None
    return _resolve(date_int, reg, path, df0, s0, kcs, kps, kcl, kpl, credit,
                    entry)


def _spread_pay(s, k_short, k_long, is_call):
    """Intrinsic we owe on a short vertical at settlement (points, >=0)."""
    if is_call:
        return max(s - k_short, 0.0) - max(s - k_long, 0.0)
    return max(k_short - s, 0.0) - max(k_long - s, 0.0)


def _resolve(date_int, reg, path, df0, s0, kcs, kps, kcl, kpl, credit, entry):
    touch = _first_touch(path, entry, kcs, kps)
    legs_in = 4
    if touch is None:                       # settle at close
        s_close = float(path["spx"].iloc[-1])
        pay = _spread_pay(s_close, kcs, kcl, True) + \
            _spread_pay(s_close, kps, kpl, False)
        pnl_pts, exit_kind, legs = credit - pay, "settle", legs_in
    else:                                   # conservative full buy-back
        bc = _quote_after(df0, touch, kcs, "C")
        bp = _quote_after(df0, touch, kps, "P")
        lc = _quote_after(df0, touch, kcl, "C")
        lp = _quote_after(df0, touch, kpl, "P")
        if None in (bc, bp, lc, lp):        # fall back to intrinsic at touch
            st = float(path[path["ms"] == touch]["spx"].iloc[0])
            cost = _spread_pay(st, kcs, kcl, True) + _spread_pay(st, kps, kpl, False)
        else:                               # buy shorts at ask, sell longs at bid
            cost = (bc[1] - lc[0]) + (bp[1] - lp[0])
        pnl_pts, exit_kind, legs = credit - cost, "stop", legs_in * 2
    commission = legs * config.COMMISSION_PER_LEG
    pnl_usd = pnl_pts * config.SPX_MULTIPLIER - commission
    width = config.H1_WING_PTS
    return {"date": date_int, "leg": "H1_pin", "regime": reg["regime"],
            "gex": reg["gex"], "entry_spx": s0, "k_call_short": kcs,
            "k_put_short": kps, "credit_pts": credit, "exit": exit_kind,
            "pnl_pts": pnl_pts, "pnl_usd": pnl_usd,
            "r_multiple": pnl_pts / width}
