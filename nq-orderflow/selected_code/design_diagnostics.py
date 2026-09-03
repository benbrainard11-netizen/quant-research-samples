"""nq_sweep_absorption_v0 — PRE-LOCK DESIGN diagnostics (DESIGN only; OOS never read).

Produces ONLY the allowed pre-lock outputs (PROTOCOL.md v2): resolved/dropped counts, class balance,
event-days/trade-days, collinearity diagnostic, out-of-fold PAIRED Δ-rank-IC MDE, coverage/drop report, and a
params.json candidate + hash. NO OOS read. NO exploratory return tables. The frozen event/target/horizon
definition is taken as-is from the protocol and is NOT changed by any result here.

Run:  & backend/.venv/Scripts/python.exe -m experiments.nq_sweep_absorption_v0.design_diagnostics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252; report uses Δ
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.nq_sweep_absorption_v0.detection_audit import (  # noqa: E402
    CONT_TICKS, DESIGN_END, DESIGN_START, GAP_MS, OPEN_MIN, OUT_DIR, REC_TICKS, REC_VOL,
    RECLAIM_MIN, RTH_LAST, TICK, bars_by_date, list_design_dates, read_day_trades,
)

OOS_DAYS_EST = 85          # OOS (2026-02-01..2026-06-09) expected usable days — ESTIMATE; OOS is never read.
B_BOOT = 2000
SEED = 12345
NS = 1_000_000_000
OFI_WINDOWS_S = (1, 5, 30)
BASE_FEATS = ["sv_1s", "sv_5s", "sv_30s", "tob_imb", "depth_total", "spread", "tod"]
BASE_INTERACT = [("sv_1s", "tob_imb"), ("sv_30s", "depth_total"), ("spread", "tod")]
# v2.1: dropped signed_agg_vol (|corr| 0.75 vs baseline) + swept_side_depth (VIF 7.4) per the collinearity finding.
SWEEP_FEATS = ["ticks_swept", "speed", "vol_per_tick", "fresh", "refill"]
SWEEP_INTERACT = [("ticks_swept", "vol_per_tick"), ("speed", "refill")]


# ---------- per-day feature + target build (structural no-lookahead) ----------
def _runs(sign, price, ts):
    brk = (sign[1:] != sign[:-1]) | ((ts[1:] - ts[:-1]) / 1e6 > GAP_MS)
    rid = np.concatenate([[0], np.cumsum(brk)])
    starts = np.flatnonzero(np.concatenate([[True], rid[1:] != rid[:-1]]))
    ends = np.concatenate([starts[1:] - 1, [len(rid) - 1]])
    return starts, ends


def day_arrays(tr: pd.DataFrame) -> dict:
    price = tr["price"].to_numpy("f8"); ts = tr["ts_ns"].to_numpy("int64")
    size = tr["size"].to_numpy("f8"); minute = tr["minute_idx"].to_numpy("int64")
    sign = np.where(tr["side"].to_numpy() == "B", 1.0, -1.0)
    bid_sz = tr["bid_sz"].to_numpy("f8"); ask_sz = tr["ask_sz"].to_numpy("f8")
    bid_px = tr["bid_px"].to_numpy("f8"); ask_px = tr["ask_px"].to_numpy("f8")
    cummax = np.maximum.accumulate(price); cummin = np.minimum.accumulate(price)
    return {"price": price, "ts": ts, "size": size, "minute": minute, "sign": sign,
            "bid_sz": bid_sz, "ask_sz": ask_sz, "bid_px": bid_px, "ask_px": ask_px, "signed_vol": sign * size,
            "prior_high": np.concatenate([[price[0]], cummax[:-1]]),
            "prior_low": np.concatenate([[price[0]], cummin[:-1]])}


def build_features(a: dict, s: int, e: int) -> dict:
    """Features for a run [s,e]. Reads ONLY indices <= e (i.e. ts <= t*). No-lookahead by construction."""
    ts = a["ts"]; price = a["price"]; sg = a["sign"][s]; p0 = price[s]; p1 = price[e]; up = sg > 0
    tstar = ts[e]; m = int(a["minute"][e]); vol = a["size"][s:e + 1].sum()
    feat = {}
    for w in OFI_WINDOWS_S:
        lo = np.searchsorted(ts, tstar - w * NS, "left")
        feat[f"sv_{w}s"] = a["signed_vol"][lo:e + 1].sum()
    tot = a["bid_sz"][e] + a["ask_sz"][e]
    feat["tob_imb"] = (a["bid_sz"][e] - a["ask_sz"][e]) / tot if tot > 0 else 0.0
    feat["depth_total"] = tot
    feat["spread"] = a["ask_px"][e] - a["bid_px"][e]
    feat["tod"] = m / RTH_LAST
    feat["ticks_swept"] = round(abs(p1 - p0) / TICK)
    feat["signed_agg_vol"] = sg * vol
    feat["speed"] = vol / max((ts[e] - ts[s]) / NS, 1e-3)
    feat["swept_side_depth"] = a["ask_sz"][e] if up else a["bid_sz"][e]
    feat["vol_per_tick"] = vol / max(feat["ticks_swept"], 1)
    feat["fresh"] = float(p1 > a["prior_high"][s] if up else p1 < a["prior_low"][s])
    opp_end = a["bid_sz"][e] if up else a["ask_sz"][e]
    opp_start = a["bid_sz"][s] if up else a["ask_sz"][s]
    feat["refill"] = opp_end - opp_start
    return feat


def resolve_target(a: dict, e: int, p1: float, sg: float):
    """Trade-level first-touch (symmetric ±CONT_TICKS from p1). Reads ONLY indices > e (ts > t*).

    Returns (label, fill_conf) or (None, 'unresolved'). stop-wins-ties (same-instant -> reversal).
    """
    ts = a["ts"]; price = a["price"]; up = sg > 0; K = CONT_TICKS * TICK
    cont_lvl = p1 + K if up else p1 - K
    rev_lvl = p1 - K if up else p1 + K
    cap = ts[e] + RECLAIM_MIN * 60 * NS
    j0 = e + 1; j1 = np.searchsorted(ts, cap, "right")
    fp = price[j0:j1]; ft = ts[j0:j1]
    ci = np.flatnonzero(fp >= cont_lvl) if up else np.flatnonzero(fp <= cont_lvl)
    ri = np.flatnonzero(fp <= rev_lvl) if up else np.flatnonzero(fp >= rev_lvl)
    tc = ft[ci[0]] if ci.size else None
    trv = ft[ri[0]] if ri.size else None
    if tc is None and trv is None:
        return None, "unresolved"
    if tc is not None and (trv is None or tc < trv):
        return 1, "exact"
    if trv is not None and (tc is None or trv < tc):
        return 0, "exact"
    return 0, "tie_reversal"


def process_day(date_int: int) -> tuple[list[dict], dict]:
    drop = {"window_miss": 0, "unresolved": 0}
    tr = read_day_trades(date_int)
    if tr.empty:
        return [], {"reason": "no_trades", **drop}
    a = day_arrays(tr)
    starts, ends = _runs(a["sign"], a["price"], a["ts"])
    rows = []
    for s, e in zip(starts, ends):
        sg = a["sign"][s]; p0 = a["price"][s]; p1 = a["price"][e]
        if (p1 - p0) * sg < REC_TICKS * TICK or a["size"][s:e + 1].sum() < REC_VOL:
            continue
        if int(a["minute"][e]) + RECLAIM_MIN > RTH_LAST:    # 15m window can't fit before 15:59 -> drop
            drop["window_miss"] += 1
            continue
        label, fconf = resolve_target(a, e, p1, sg)         # reads only ts > t*
        if label is None:
            drop["unresolved"] += 1
            continue
        feat = build_features(a, s, e)                       # reads only ts <= t*
        feat.update({"et_date": date_int, "label": label, "fill_conf": fconf})
        rows.append(feat)
    return rows, {"reason": "ok", **drop}


# ---------- design matrices ----------
def _design(df: pd.DataFrame, cols: list[str], inter: list[tuple]) -> tuple[np.ndarray, list[str]]:
    names = list(cols)
    mats = [df[cols].to_numpy("f8")]
    for a, b in inter:
        mats.append((df[a] * df[b]).to_numpy("f8").reshape(-1, 1)); names.append(f"{a}*{b}")
    return np.column_stack(mats), names


def _standardize(fit: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = fit.mean(0); sd = fit.std(0); sd[sd == 0] = 1.0
    return (fit - mu) / sd, (apply - mu) / sd


def _fit_predict(Xtr, ytr, Xev):
    m = LogisticRegression(penalty=None, max_iter=2000)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xev)[:, 1]


# ---------- collinearity ----------
def collinearity(df: pd.DataFrame) -> dict:
    Xb, _ = _design(df, BASE_FEATS, BASE_INTERACT)
    Xb = (Xb - Xb.mean(0)) / np.where(Xb.std(0) == 0, 1, Xb.std(0))
    Xb1 = np.column_stack([np.ones(len(Xb)), Xb])
    y = df["label"].to_numpy("f8"); ry = y - Xb1 @ np.linalg.lstsq(Xb1, y, rcond=None)[0]
    out = {}
    for f in SWEEP_FEATS:
        v = df[f].to_numpy("f8"); v = (v - v.mean()) / (v.std() or 1)
        beta = np.linalg.lstsq(Xb1, v, rcond=None)[0]
        r2 = 1 - ((v - Xb1 @ beta) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12)
        rf = v - Xb1 @ beta
        maxcorr = max(abs(np.corrcoef(v, Xb[:, j])[0, 1]) for j in range(Xb.shape[1]))
        pc = np.corrcoef(rf, ry)[0, 1] if rf.std() > 0 and ry.std() > 0 else 0.0
        out[f] = {"vif": float(1 / max(1 - r2, 1e-6)), "max_abs_corr_baseline": float(maxcorr),
                  "partial_corr_label": float(pc)}
    return out


# ---------- out-of-fold paired Δ rank-IC MDE ----------
def oof_mde(df: pd.DataFrame) -> dict:
    days = np.sort(df["et_date"].unique())
    cut = days[len(days) // 2]
    tr = df[df["et_date"] < cut]; ev = df[df["et_date"] >= cut]
    Xb_tr, _ = _design(tr, BASE_FEATS, BASE_INTERACT); Xb_ev, _ = _design(ev, BASE_FEATS, BASE_INTERACT)
    Xa_tr, _ = _design(tr, BASE_FEATS + SWEEP_FEATS, BASE_INTERACT + SWEEP_INTERACT)
    Xa_ev, _ = _design(ev, BASE_FEATS + SWEEP_FEATS, BASE_INTERACT + SWEEP_INTERACT)
    Xb_tr, Xb_ev = _standardize(Xb_tr, Xb_ev); Xa_tr, Xa_ev = _standardize(Xa_tr, Xa_ev)
    ytr = tr["label"].to_numpy(); yev = ev["label"].to_numpy()
    pb = _fit_predict(Xb_tr, ytr, Xb_ev); pa = _fit_predict(Xa_tr, ytr, Xa_ev)
    ed = ev["et_date"].to_numpy()
    point = spearmanr(pa, yev).statistic - spearmanr(pb, yev).statistic
    uniq = np.unique(ed); rng = np.random.default_rng(SEED); deltas = np.empty(B_BOOT)
    by = {d: np.flatnonzero(ed == d) for d in uniq}
    for b in range(B_BOOT):
        idx = np.concatenate([by[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        deltas[b] = spearmanr(pa[idx], yev[idx]).statistic - spearmanr(pb[idx], yev[idx]).statistic
    sd = float(deltas.std())
    factor = (len(uniq) / OOS_DAYS_EST) ** 0.5
    return {"n_eval": int(len(ev)), "n_eval_days": int(len(uniq)), "oof_delta_ic_point": float(point),
            "mde95_eval": float(1.96 * sd), "mde95_oos_scaled": float(1.96 * sd * factor),
            "ci_lo_pctile_eval": float(np.percentile(deltas, 2.5)), "n_oos_days_est": OOS_DAYS_EST,
            "B": B_BOOT, "seed": SEED}


def icc(df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy("f8"); g = df.groupby("et_date")["label"]
    k = g.ngroups; N = len(y); ni = g.count().to_numpy("f8"); mi = g.mean().to_numpy("f8")
    gm = y.mean()
    ssb = float((ni * (mi - gm) ** 2).sum()); ssw = float(((y - df["et_date"].map(g.mean()).to_numpy()) ** 2).sum())
    msb = ssb / max(k - 1, 1); msw = ssw / max(N - k, 1)
    n0 = (N - (ni ** 2).sum() / N) / max(k - 1, 1)
    rho = (msb - msw) / max(msb + (n0 - 1) * msw, 1e-12)
    rho = float(min(max(rho, 0.0), 1.0))
    deff = 1 + (n0 - 1) * rho
    return {"icc": rho, "effective_n": float(N / deff), "raw_n": N, "mean_per_day": float(N / k)}


# ---------- driver ----------
def run(quick: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = list_design_dates(DESIGN_START, DESIGN_END)
    if quick:
        dates = dates[:quick]
    bars = bars_by_date(DESIGN_START, DESIGN_END)
    rows, drops, usable, event_days = [], {"no_trades": 0, "partial_or_missing_1m": 0, "window_miss": 0, "unresolved": 0}, 0, 0
    for i, d in enumerate(dates):
        if d not in bars:
            drops["partial_or_missing_1m"] += 1
            continue
        day_rows, info = process_day(d)
        if info["reason"] == "no_trades":
            drops["no_trades"] += 1
            continue
        usable += 1
        drops["window_miss"] += info["window_miss"]; drops["unresolved"] += info["unresolved"]
        if day_rows:
            event_days += 1
            rows.extend(day_rows)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(dates)} days, events so far={len(rows)}")
    df = pd.DataFrame(rows)
    finite = np.isfinite(df[BASE_FEATS + SWEEP_FEATS].to_numpy("f8")).all(1)   # single joint mask
    nonfin = int((~finite).sum()); df = df[finite].reset_index(drop=True)
    qualifying = len(rows) + drops["window_miss"] + drops["unresolved"]
    _emit(df, drops, usable, event_days, qualifying, nonfin, len(dates))


def _emit(df, drops, usable, event_days, qualifying, nonfin, n_dates):
    bal = float(df["label"].mean()); col = collinearity(df); mde = oof_mde(df); ic = icc(df)
    resolved = len(df)
    ties = int((df["fill_conf"] == "tie_reversal").sum())
    cov = resolved / max(qualifying, 1)
    par = {
        "spec_frozen": {
            "event": {"min_ticks": REC_TICKS, "min_vol": REC_VOL, "gap_ms": GAP_MS, "tick": TICK,
                      "decision_time": "sweep last-trade ts", "fresh_anchor": "RTH-open running extreme prior to run start",
                      "open_window_min": OPEN_MIN},
            "target": {"continuation": f"p1 +{CONT_TICKS}t (sweep dir)",
                       "reversal": f"p1 -{CONT_TICKS}t (against) — SYMMETRIC, size-decoupled (v2.1)",
                       "cap_min": RECLAIM_MIN, "resolution": "trade-level first-touch", "tie": "reversal (stop-wins-ties)",
                       "drop": "window-miss + unresolved-in-cap",
                       "fix_note": "v2.1 removed the origin-p0 reversal barrier (it = ticks_swept distance -> tautology)"},
            "baseline_features": BASE_FEATS, "baseline_interactions": [f"{a}*{b}" for a, b in BASE_INTERACT],
            "sweep_features": SWEEP_FEATS, "sweep_interactions": [f"{a}*{b}" for a, b in SWEEP_INTERACT],
            "ofi_note": "MBP-1 top-of-book: OFI proxied by signed-trade-volume windows; depth-imbalance==TOB-imbalance so listed once (no duplicate columns).",
            "model": "logistic (penalty=None), frozen on DESIGN, row-independent on OOS; single joint finite mask; paired day-block bootstrap",
            "split": {"design": [DESIGN_START, DESIGN_END], "oos": [20260201, 20260609], "oos_sealed": True},
            "primary": "Delta rank-IC = rankIC(p_aug,y) - rankIC(p_base,y), paired day-block bootstrap",
            "gate": "one-sided lower 97.5% percentile CI of Delta > 0 AND Delta >= MDE_frozen AND clears nulls+random-feature AND day-balanced same-sign AND both-half point>=0",
        },
        "design_diagnostics": {
            "dates_scanned": n_dates, "usable_trade_days": usable, "event_days": event_days,
            "qualifying_events": qualifying, "resolved_events": resolved,
            "dropped": drops, "nonfinite_feature_rows_dropped": nonfin,
            "same_instant_ties_to_reversal": ties,
            "coverage_resolved_over_qualifying": cov,
            "class_balance_continuation_rate": bal,
            "within_day": ic, "collinearity": col, "mde": mde,
        },
        "NOTE": "CANDIDATE for lock review only. NOT locked. OOS never read. Frozen def unchanged by these results.",
    }
    blob = json.dumps(par, sort_keys=True, separators=(",", ":")).encode()
    par["sha256"] = hashlib.sha256(blob).hexdigest()
    (OUT_DIR / "params_candidate.json").write_text(json.dumps(par, indent=2), encoding="utf-8")

    L = ["# nq_sweep_absorption_v0 — PRE-LOCK DESIGN diagnostics (DESIGN only; OOS NOT read)\n",
         "> Allowed pre-lock outputs only. No exploratory return tables. Frozen def unchanged by results.\n",
         f"- usable trade-days: {usable} | event-days: {event_days} | dates scanned: {n_dates}",
         f"- qualifying events: {qualifying:,} | resolved: {resolved:,} | coverage: {cov*100:.1f}%",
         f"- dropped: window_miss={drops['window_miss']:,} unresolved={drops['unresolved']:,} "
         f"no_trades={drops['no_trades']} partial_1m={drops['partial_or_missing_1m']} nonfinite={nonfin}",
         f"- same-instant ties -> reversal: {ties:,} ({ties/max(resolved,1)*100:.2f}% of resolved)",
         f"- class balance (continuation rate): {bal*100:.1f}%  (reversal {100-bal*100:.1f}%)",
         f"- within-day ICC(label): {ic['icc']:.3f} | raw n {ic['raw_n']:,} -> effective n {ic['effective_n']:.0f} "
         f"(mean {ic['mean_per_day']:.0f}/day)",
         "\n## Collinearity of each sweep feature vs the frozen baseline (DESIGN)",
         "| feature | VIF | max|corr| baseline | partial-corr w/ label |", "|---|---:|---:|---:|"]
    for f in SWEEP_FEATS:
        c = col[f]
        L.append(f"| {f} | {c['vif']:.2f} | {c['max_abs_corr_baseline']:.3f} | {c['partial_corr_label']:+.4f} |")
    L += ["\n## Out-of-fold PAIRED Δ rank-IC MDE (feasibility; DESIGN halves; NOT the OOS verdict)",
          f"- eval n {mde['n_eval']:,} over {mde['n_eval_days']} days | OOF Δ-IC point {mde['oof_delta_ic_point']:+.4f} "
          f"(feasibility only)",
          f"- MDE95 (eval-window) {mde['mde95_eval']:.4f} | **MDE95 scaled to OOS ~{mde['n_oos_days_est']}d "
          f"{mde['mde95_oos_scaled']:.4f}** | 2.5th-pctile {mde['ci_lo_pctile_eval']:+.4f} | B={mde['B']}",
          f"\nparams_candidate.json sha256: `{par['sha256']}`",
          "\n_For GPT/Ben lock review. Not locked. Run did not read OOS._"]
    (OUT_DIR / "DESIGN_DIAGNOSTICS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", type=int, default=None)
    run(p.parse_args(argv).quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
