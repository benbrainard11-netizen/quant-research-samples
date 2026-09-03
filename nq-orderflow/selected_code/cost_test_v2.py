"""nq_sweep_absorption_v0 — COST-TEST v2: does PASSIVE (maker) entry + move-size selection rescue the edge?

v1 verdict was NO: the naive TAKER ±8t trade loses to frictions (best gross +0.70t, net negative at 2t). v2 asks the
two open levers (fill-model design panel `costtest-v2-fillmodel-design`): (1) STOP paying the spread — passive limit
entry, filled HONESTLY against the real MBP-1 tick path; (2) trade only big-expected-move events (top-atr_30 tercile).

The honest fill model (where fake edges hide — every rule chosen to make a fake edge HARDER):
  - limit L = R − bet_dir·OFFSET·tick (R = first post-t* print, v1's entry). LONG posts below, SHORT above.
  - FILL = strict trade-THROUGH on the correct aggressor side (a print AT L does NOT fill — you sit behind an unseen
    queue; MBP-1 shows no cancels). q0haircut (join back of resting queue Q0=bid/ask_sz[e]) is the optimistic bookend.
  - ADVERSE barrier checked BEFORE fill each tick; same-tick tie → barrier wins → MISS (the runner got away = the
    adverse selection; never dropped). EXIT is a TAKER ±8t stop/target (the exit is never passive).
  - MISS-INCLUSIVE accounting: a miss falls back to v1's taker P&L. So maker beats v1 ONLY if it wins on the events it
    actually fills passively. Headline = miss-inclusive net over the IDENTICAL v1 universe.
Frictions: maker FILLED = maker_gross − exit_taker_cost {1.0,1.5,2.0} (headline 1.5); MISS charged v1 taker 2.0t.
DESIGN-OOF / development only — NOT a verdict; sealed/forward OOS preserved. NQ $5/tick.
  & backend/.venv/Scripts/python.exe -m experiments.nq_sweep_absorption_v0.cost_test_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.nq_sweep_absorption_v0.detection_audit import (  # noqa: E402
    CONT_TICKS, DESIGN_START, DESIGN_END, RECLAIM_MIN, REC_TICKS, REC_VOL, RTH_LAST, TICK,
    bars_by_date, list_design_dates, read_day_trades)
from experiments.nq_sweep_absorption_v0.design_diagnostics import (  # noqa: E402
    NS, day_arrays, resolve_target, _runs, build_features)
from experiments.nq_sweep_absorption_v0.oos_read import ens, AUG  # noqa: E402
from experiments.nq_sweep_absorption_v0.cost_test import oof_score  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "out"
TICK_VALUE = 5.0
OFFSETS, MODES = [1, 2], ["through", "q0haircut", "touch"]      # touch = BANNED headline; measured only for the gap
EXIT_FRICTIONS, HEAD_EXIT, TAKER_FR = [1.0, 1.5, 2.0], 1.5, 2.0
CONFIGS = [(1, "through"), (1, "q0haircut"), (1, "touch"), (2, "through")]   # (offset, mode) precomputed per bet_dir
B_BOOT, SEED = 2000, 12345


def _resolve_from(a, jf, p1, sg, cap_ns):
    """TAKER ±8t first-touch exit STRICTLY after fill index jf (same geometry/tie as resolve_target). Time-exit→mid."""
    ts, price = a["ts"], a["price"]; up = sg > 0; K = CONT_TICKS * TICK
    cont_lvl = p1 + K if up else p1 - K; rev_lvl = p1 - K if up else p1 + K
    j1 = int(np.searchsorted(ts, cap_ns, "right"))
    if jf >= j1:
        k = min(jf, len(price) - 1); return (a["bid_px"][k] + a["ask_px"][k]) / 2.0
    fp = price[jf:j1]                                                # INCLUSIVE of the fill tick (matches v1's e+1 scan)
    ci = np.flatnonzero(fp >= cont_lvl) if up else np.flatnonzero(fp <= cont_lvl)
    ri = np.flatnonzero(fp <= rev_lvl) if up else np.flatnonzero(fp >= rev_lvl)
    tc = ci[0] if ci.size else 10**9; trv = ri[0] if ri.size else 10**9
    if tc == 10**9 and trv == 10**9:
        return (a["bid_px"][j1 - 1] + a["ask_px"][j1 - 1]) / 2.0     # time-exit at cap (taker liquidation at mid)
    return rev_lvl if trv <= tc else cont_lvl                        # stop-wins-ties → reversal


def maker_outcome(a, e, p1, sg, bet_dir, offset, mode, cap_ns):
    """(filled?, gross_ticks). Adverse-barrier-before-fill; strict trade-through (or q0/touch). gross = bet_dir·(exit−L)/t."""
    ts, price, sign, size = a["ts"], a["price"], a["sign"], a["size"]
    if e + 1 >= len(price):
        return False, np.nan
    R = price[e + 1]; L = round((R - bet_dir * offset * TICK) / TICK) * TICK; Kp = CONT_TICKS * TICK
    j1 = int(np.searchsorted(ts, cap_ns, "right"))
    if j1 <= e + 1:
        return False, np.nan
    fp = price[e + 1:j1]; fs = sign[e + 1:j1]; fz = size[e + 1:j1]
    dom = (fp - p1) * sg                                             # signed progress in the sweep direction
    bh = np.flatnonzero((dom >= Kp) | (dom <= -Kp))                 # EITHER ±8t barrier resolves the H1 trade
    first_barrier = bh[0] if bh.size else 10**9                     # a barrier hit BEFORE you fill = no position (MISS)
    if mode == "through":
        fi = np.flatnonzero((fs == -1) & (fp < L)) if bet_dir > 0 else np.flatnonzero((fs == 1) & (fp > L))
        first_fill = fi[0] if fi.size else 10**9
    elif mode == "touch":                                            # BANNED headline: at-or-through, no queue
        fi = np.flatnonzero(fp <= L) if bet_dir > 0 else np.flatnonzero(fp >= L)
        first_fill = fi[0] if fi.size else 10**9
    else:                                                            # q0haircut: clear resting queue Q0, then fill at L
        Q0 = a["bid_sz"][e] if bet_dir > 0 else a["ask_sz"][e]
        atL = (fp == L) & (fs == -1) if bet_dir > 0 else (fp == L) & (fs == 1)
        thr = np.flatnonzero((fs == -1) & (fp < L)) if bet_dir > 0 else np.flatnonzero((fs == 1) & (fp > L))
        cum = np.cumsum(fz * atL); qf = np.flatnonzero(atL & (cum >= Q0))
        cand = [x[0] for x in (qf, thr) if x.size]
        first_fill = min(cand) if cand else 10**9
    if first_fill >= first_barrier:                                  # a ±8t barrier resolved first (or tie/no fill) → MISS
        return False, np.nan
    jf = e + 1 + int(first_fill)
    return True, float(bet_dir * (_resolve_from(a, jf, p1, sg, cap_ns) - L) / TICK)


def build_events(dates):
    bars = bars_by_date(min(dates), max(dates))
    rows = []
    for i, d in enumerate(dates):
        if d not in bars:
            continue
        tr = read_day_trades(d)
        if tr.empty:
            continue
        a = day_arrays(tr); price = a["price"]; n = len(price); b = bars[d]
        hi, lo = b["high"], b["low"]
        starts, ends = _runs(a["sign"], a["price"], a["ts"])
        for s, e in zip(starts, ends):
            sg = a["sign"][s]; p0 = price[s]; p1 = price[e]
            if (p1 - p0) * sg < REC_TICKS * TICK or a["size"][s:e + 1].sum() < REC_VOL:
                continue
            if int(a["minute"][e]) + RECLAIM_MIN > RTH_LAST:
                continue
            label, _ = resolve_target(a, e, p1, sg)
            if label is None or e + 1 >= n:
                continue
            m = int(a["minute"][e])
            f = build_features(a, s, e)
            entry = price[e + 1]
            cont_barrier = p1 + CONT_TICKS * TICK if sg > 0 else p1 - CONT_TICKS * TICK
            rev_barrier = p1 - CONT_TICKS * TICK if sg > 0 else p1 + CONT_TICKS * TICK
            exit_barrier = cont_barrier if label == 1 else rev_barrier
            pnl_cont = sg * (exit_barrier - entry) / TICK              # v1 continuation-bet taker P&L
            cap_ns = a["ts"][e] + RECLAIM_MIN * 60 * NS
            row = {**{k: f[k] for k in AUG}, "et_date": int(d), "label": int(label), "sg": float(sg),
                   "pnl_cont": float(pnl_cont), "atr_30": (float(np.mean(hi[m - 30:m] - lo[m - 30:m])) if m >= 30 else np.nan)}
            for bd in (1, -1):
                tag = "p1" if bd > 0 else "m1"
                for off, mode in CONFIGS:
                    fl, g = maker_outcome(a, e, p1, sg, bd, off, mode, cap_ns)
                    row[f"fill_{tag}_o{off}_{mode[:3]}"] = bool(fl); row[f"g_{tag}_o{off}_{mode[:3]}"] = g
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(dates)} days, events={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    return df[np.isfinite(df[AUG].to_numpy("f8")).all(1)].reset_index(drop=True)


def _pick(df, bet_dir, off, mode):
    """Per-event (fill, gross) for the event's own bet_dir (vectorized select of the p1/m1 precomputed columns)."""
    pos = bet_dir > 0
    fl = np.where(pos, df[f"fill_p1_o{off}_{mode[:3]}"], df[f"fill_m1_o{off}_{mode[:3]}"])
    g = np.where(pos, df[f"g_p1_o{off}_{mode[:3]}"], df[f"g_m1_o{off}_{mode[:3]}"])
    return fl.astype(bool), g.astype("f8")


def _dayblock_ci(vals, days):
    uniq = np.unique(days); by = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(SEED)
    m = np.array([vals[np.concatenate([by[d] for d in rng.choice(uniq, len(uniq), True)])].mean() for _ in range(B_BOOT)])
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _cell(df, bet_dir, off, mode, exit_fr, mask=None):
    """Miss-inclusive net over the (masked) universe: filled → gross−exit_fr; miss → v1 taker P&L − 2.0t."""
    if mask is None:
        mask = np.ones(len(df), bool)
    sub = df[mask]; days = sub["et_date"].to_numpy(); bd = bet_dir[mask]
    fl, g = _pick(sub, bd, off, mode)
    wbt = bd * sub["sg"].to_numpy("f8") * sub["pnl_cont"].to_numpy("f8")      # would-be v1 taker P&L for this bet
    payoff = np.where(fl, g - exit_fr, wbt - TAKER_FR)
    mean, lo, hi = _dayblock_ci(payoff, days)
    nd = len(np.unique(days))
    return {"n": int(len(sub)), "trades_per_day": round(len(sub) / max(nd, 1), 1), "fill_rate_pct": round(float(fl.mean()) * 100, 1),
            "miss_incl_net_t": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)], "ci_lo_gt0": bool(lo > 0),
            "filled_only_net_t": round(float((g[fl] - exit_fr).mean()), 3) if fl.any() else None,
            "net_$_per_trade": round(mean * TICK_VALUE, 2), "net_$_per_day": round(float(pd.Series(payoff).groupby(pd.Series(days)).sum().mean()) * TICK_VALUE, 0)}


def _v1_cell(df, bet_dir, fr, mask=None):
    if mask is None:
        mask = np.ones(len(df), bool)
    sub = df[mask]; days = sub["et_date"].to_numpy(); bd = bet_dir[mask]
    p = bd * sub["sg"].to_numpy("f8") * sub["pnl_cont"].to_numpy("f8") - fr
    mean, lo, hi = _dayblock_ci(p, days)
    return {"n": int(len(sub)), "net_t": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)], "ci_lo_gt0": bool(lo > 0)}


def _assert_degeneracy(dates):
    """Integrity: where OFFSET=0/touch FILLS, its P&L must equal v1 taker exactly; and EVERY miss must be explained by the
    adverse barrier already breached at e+1 (first post-t* print ≥8t past p1 = instant-resolve gap). A miss without a breach
    = a fill-logic bug → abort."""
    K = CONT_TICKS * TICK; diffs, gap_miss, bad_miss, checked = [], 0, 0, 0
    for d in dates[:5]:
        tr = read_day_trades(d)
        if tr.empty:
            continue
        a = day_arrays(tr); price = a["price"]; n = len(price)
        starts, ends = _runs(a["sign"], a["price"], a["ts"])
        for s, e in zip(starts, ends):
            sg = a["sign"][s]; p0 = price[s]; p1 = price[e]
            if (p1 - p0) * sg < REC_TICKS * TICK or a["size"][s:e + 1].sum() < REC_VOL or int(a["minute"][e]) + RECLAIM_MIN > RTH_LAST:
                continue
            label, _ = resolve_target(a, e, p1, sg)
            if label is None or e + 1 >= n:
                continue
            entry = price[e + 1]; cb = p1 + K if sg > 0 else p1 - K; rb = p1 - K if sg > 0 else p1 + K
            pnl_cont = sg * ((cb if label == 1 else rb) - entry) / TICK
            cap = a["ts"][e] + RECLAIM_MIN * 60 * NS
            for bd in (1, -1):
                fl, g = maker_outcome(a, e, p1, sg, bd, 0, "touch", cap)
                if fl:
                    diffs.append(abs(g - bd * sg * pnl_cont))
                elif abs(entry - p1) >= K - 1e-9:                          # a ±8t barrier already resolved at e+1 (gap)
                    gap_miss += 1
                else:
                    bad_miss += 1
            checked += 1
        if checked >= 200:
            break
    exact = float(np.mean(np.array(diffs) < 1e-6)) if diffs else 0.0
    assert checked and bad_miss == 0 and exact >= 0.999, \
        f"degeneracy fail: filled-exact {exact:.1%} of {len(diffs)}, bad_miss {bad_miss} (fill-logic bug)"
    print(f"degeneracy ✅ OFFSET=0/touch = v1 on {exact:.1%} of {len(diffs)} filled; {gap_miss} instant-resolve gap-misses; 0 bad", flush=True)


def run(quick=None):
    OUT.mkdir(parents=True, exist_ok=True)
    dates = list_design_dates(DESIGN_START, DESIGN_END)
    if quick:
        dates = dates[:quick]; print(f"SMOKE: first {quick} DESIGN dates (non-representative)", flush=True)
    _assert_degeneracy(dates)
    print("building H1 events + honest passive-fill tick walks (re-reads MBP-1)…", flush=True)
    df = build_events(dates)
    score = oof_score(df); sg = df["sg"].to_numpy("f8")
    fade = -sg; cont = sg; h1 = np.where(score > 0.5, sg, -sg)        # bet_dir per strategy
    rng = np.random.default_rng(SEED); rand = rng.choice([-1.0, 1.0], len(df))
    atr = df["atr_30"].to_numpy("f8"); have = np.isfinite(atr)
    top_terc = have & (atr >= np.nanquantile(atr, 2 / 3))

    cells = {
        "PRIMARY_fade_through_o1_all_exit1.5": _cell(df, fade, 1, "through", HEAD_EXIT),
        "fade_through_o1_all_FILLEDvsMISS_gap": {"filled_only": _cell(df, fade, 1, "through", HEAD_EXIT)["filled_only_net_t"]},
        "fade_through_o2_all_exit1.5": _cell(df, fade, 2, "through", HEAD_EXIT),
        "fade_q0haircut_o1_all_exit1.5_OPTIMISTIC": _cell(df, fade, 1, "q0haircut", HEAD_EXIT),
        "fade_through_o1_TOPATRtercile_exit1.5": _cell(df, fade, 1, "through", HEAD_EXIT, top_terc),
        "h1directed_through_o1_all_exit1.5_secondary": _cell(df, h1, 1, "through", HEAD_EXIT),
        "v1_TAKER_fade_all_2.0t_THE_BAR": _v1_cell(df, fade, TAKER_FR),
        "exit_friction_sensitivity_primary": {f"exit_{fr:g}t": _cell(df, fade, 1, "through", fr) for fr in EXIT_FRICTIONS},
    }
    # fill-rate-matched v1 (the filled subset only) — does maker beat taker on the SAME events it fills?
    fl_primary, _ = _pick(df, fade, 1, "through")
    cells["v1_TAKER_fade_on_maker_FILLED_subset_2.0t"] = _v1_cell(df, fade, TAKER_FR, fl_primary)

    # ---- placebos (guard against a FALSE go) ----
    g_thr = _cell(df, fade, 1, "through", HEAD_EXIT)["miss_incl_net_t"]
    g_tch = _cell(df, fade, 1, "touch", HEAD_EXIT)["miss_incl_net_t"]
    fl_f, _g = _pick(df, fade, 1, "through"); lab = df["label"].to_numpy()
    placebos = {
        "random_direction_miss_incl_net_t": _cell(df, rand, 1, "through", HEAD_EXIT)["miss_incl_net_t"],
        "at_touch_vs_through_gap_t": round(g_tch - g_thr, 3),
        "primary_filled_only_minus_miss_incl_t": round(cells["PRIMARY_fade_through_o1_all_exit1.5"]["filled_only_net_t"]
                                                       - cells["PRIMARY_fade_through_o1_all_exit1.5"]["miss_incl_net_t"], 3),
        "fade_fill_rate_pct": cells["PRIMARY_fade_through_o1_all_exit1.5"]["fill_rate_pct"],
        "fade_fillrate_on_reversals_WINS_pct": round(float(fl_f[lab == 0].mean()) * 100, 1),
        "fade_fillrate_on_continuations_LOSSES_pct": round(float(fl_f[lab == 1].mean()) * 100, 1),
        "shuffled_atr_tercile_net_t": _cell(df, fade, 1, "through", HEAD_EXIT,
            have & (rng.permutation(atr) >= np.nanquantile(atr, 2 / 3)))["miss_incl_net_t"],
    }

    p = cells["PRIMARY_fade_through_o1_all_exit1.5"]; bar = cells["v1_TAKER_fade_all_2.0t_THE_BAR"]
    dvs = round(p["miss_incl_net_t"] - bar["net_t"], 3); beats_v1 = dvs > 0
    no_leak = placebos["random_direction_miss_incl_net_t"] < 0
    if p["ci_lo_gt0"] and beats_v1 and no_leak:
        verdict = (f"GO — primary miss-inclusive net {p['miss_incl_net_t']:+.2f} t clears exit-1.5t with CI_lo>0 AND beats v1 "
                   f"taker by {dvs:+.2f}; placebos clean. A maker futures strategy plausibly exists → forward OOS.")
    elif beats_v1 and p["miss_incl_net_t"] < 0:
        verdict = (f"NO, but informative — passive entry IMPROVES on the taker trade by {dvs:+.2f} t/trade (a fixed +1t entry "
                   f"on {p['fill_rate_pct']}% fills, surviving the honest trade-THROUGH model — NOT eaten by adverse selection), "
                   f"yet the ±8t fade trade stays too unprofitable ({p['miss_incl_net_t']:+.2f} t net, CI {p['ci']}) for it to "
                   f"clear 0. The entry leg is ~maxed; futures need a better BASE trade (move-size targets/selection) or options "
                   f"convexity. H1 remains a forecast, not yet a futures trade.")
    else:
        verdict = ("NO — passive entry does not even beat the taker trade: the spread saved = the adverse selection eaten "
                   "(miss-inclusive ≤ v1; the at-touch-vs-through gap is the fake-edge magnitude). H1 stays a forecast → "
                   "options convexity becomes the necessary lane.")

    nums = {"scope": "DESIGN-OOF / development only — NOT a verdict; sealed/forward OOS preserved",
            "events": int(len(df)), "design_days": int(df["et_date"].nunique()),
            "continuation_rate_pct": round(float(df["label"].mean()) * 100, 1),
            "nq_tick_value_$": TICK_VALUE, "headline_exit_friction_t": HEAD_EXIT, "taker_friction_t": TAKER_FR,
            "primary_cell": "fade · through · OFFSET=1 · all-events · miss-inclusive · exit 1.5t (vs v1 fade taker 2.0t, same universe)",
            "cells": cells, "placebos": placebos, "verdict": verdict}
    params = {"OFFSETS": OFFSETS, "MODES": MODES, "EXIT_FRICTIONS": EXIT_FRICTIONS, "HEAD_EXIT": HEAD_EXIT,
              "TAKER_FR": TAKER_FR, "primary_cell": nums["primary_cell"], "fill": "strict trade-through; miss-inclusive; taker exit",
              "selection": "top atr_30 tercile (placebo-grade, fixed K=8)", "B_BOOT": B_BOOT, "SEED": SEED}
    params["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in params.items()}, sort_keys=True).encode()).hexdigest()[:16]
    (OUT / "cost_test_v2_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    if quick:
        print(f"\nSMOKE primary {p['miss_incl_net_t']:+.3f}t ci{p['ci']} fill {p['fill_rate_pct']}% | v1 bar {bar['net_t']:+.3f} | "
              f"touch-gap {placebos['at_touch_vs_through_gap_t']:+.3f} | rand {placebos['random_direction_miss_incl_net_t']:+.3f}\n{verdict}")
        return 0
    (OUT / "cost_test_v2_numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")
    _md(nums)
    return 0


def _md(n):
    p = n["cells"]["PRIMARY_fade_through_o1_all_exit1.5"]; bar = n["cells"]["v1_TAKER_fade_all_2.0t_THE_BAR"]; pl = n["placebos"]
    L = ["# COST-TEST v2 — does PASSIVE entry + move-size selection rescue the edge? (DESIGN-OOF; NOT a verdict)\n",
         "> Honest maker fills on the real MBP-1 tick path: strict trade-THROUGH (AT-touch ≠ fill), adverse-barrier-before-",
         "> fill (miss = the runner got away), TAKER ±8t exit, MISS-INCLUSIVE accounting (a miss falls back to v1's taker",
         "> P&L). So maker beats v1 only if it wins on the events it fills passively. NQ $5/tick. DEV only — no verdict.\n",
         f"- events {n['events']:,} over {n['design_days']} days | continuation {n['continuation_rate_pct']}% → baseline = always-fade",
         f"- **PRIMARY** (fade · through · OFFSET=1 · all · miss-incl · exit 1.5t): net **{p['miss_incl_net_t']:+.3f} t** "
         f"(${p['net_$_per_trade']:+.2f}/trade) day-block CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | fill-rate {p['fill_rate_pct']}% | "
         f"filled-only {p['filled_only_net_t']:+.3f} t",
         f"- **the bar — v1 taker fade @2.0t over the same universe:** {bar['net_t']:+.3f} t (CI [{bar['ci'][0]:+.3f}, {bar['ci'][1]:+.3f}])",
         "\n## Cells (DESIGN-OOF; only the PRIMARY is decisional — the rest are pre-registered sensitivity/secondary)",
         "| cell | net t | CI | fill% | filled-only |", "|---|---:|---:|---:|---:|"]
    for k, c in n["cells"].items():
        if isinstance(c, dict) and "miss_incl_net_t" in c:
            L.append(f"| {k} | {c['miss_incl_net_t']:+.3f} | [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] | {c['fill_rate_pct']} | "
                     f"{c['filled_only_net_t']} |")
    L += ["\n## Placebos (guard against a FALSE go)",
          f"- random-direction miss-incl net: **{pl['random_direction_miss_incl_net_t']:+.3f} t** (should be ≈ −friction; >0 ⇒ execution-artifact leak)",
          f"- **at-touch − through gap: {pl['at_touch_vs_through_gap_t']:+.3f} t** (the queue-illusion / fake-edge magnitude)",
          f"- filled-only − miss-inclusive: {pl['primary_filled_only_minus_miss_incl_t']:+.3f} t (the adverse-selection magnitude — the events you passively fill are worse fade trades)",
          f"- fade fill-rate on its WINS (reversals) {pl['fade_fillrate_on_reversals_WINS_pct']}% vs its LOSSES (continuations) "
          f"{pl['fade_fillrate_on_continuations_LOSSES_pct']}% — the fade edge lives in fast reversals that never return to a passive limit",
          f"- shuffled-atr tercile net {pl['shuffled_atr_tercile_net_t']:+.3f} t (≈ all-events ⇒ selection adds nothing)",
          f"\n## Verdict (development economics — NOT a lock)\n- **{n['verdict']}**",
          "\n_DESIGN-OOF / development only. No OOS read, no lock, no live trading. The forward window is the real economic OOS._"]
    (OUT / "COST_TEST_V2.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"\nwrote {OUT/'COST_TEST_V2.md'} + cost_test_v2_numbers.json")


if __name__ == "__main__":
    _p = argparse.ArgumentParser(); _p.add_argument("--quick", type=int, default=None)
    raise SystemExit(run(_p.parse_args().quick))
