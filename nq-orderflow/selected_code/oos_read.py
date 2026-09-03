"""nq_sweep_absorption_v0 — the LOCKED v3.1 read (the firewall as code).

PRIMARY = deterministic 10-seed LightGBM ENSEMBLE (kills single-seed luck), tie-averaged rank-IC end-to-end.
Δ = rankIC(p_aug,y) − rankIC(p_base,y); M_base=ens(baseline), M_aug=ens(baseline+sweep). Paired day-block
bootstrap CI; nulls = tree-shaped CAPACITY FLOOR (permuted-sweep, decisive DoF control) + within-day label
shuffle; per-feature ablation; day-balanced; both halves. Emits ONLY these (output firewall).

  --design                : pre-lock finalize on DESIGN out-of-fold; writes params_candidate_v3.json (v3.1) + hash.
                            DESIGN ONLY — OOS untouched.
  --oos  --confirm-locked : the ONE post-lock read on the sealed OOS. REFUSES without --confirm-locked.

Null set vs the protocol's 5: capacity-floor replaces the (linear) random-feature control AND is the decisive
tree DoF null; within-day shuffle replaces day-permutation+label-shuffle (preserves daily base-rate + clustering);
feature-forward-shift is subsumed by the STRUCTURAL leak tests (tests/test_leakage.py 5/5); matched-donor-day-swap
is impractical at event-aggregation level. Documented in params.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.nq_sweep_absorption_v0.detection_audit import (  # noqa: E402
    CONT_TICKS, DESIGN_END, DESIGN_START, GAP_MS, MBP1_NQ, OOS_START, OPEN_MIN, OUT_DIR,
    RECLAIM_MIN, REC_TICKS, REC_VOL, RTH_LAST, TICK, bars_by_date, list_design_dates, read_day_trades,
)
from experiments.nq_sweep_absorption_v0.design_diagnostics import (  # noqa: E402
    BASE_FEATS, OOS_DAYS_EST, SEED, SWEEP_FEATS, build_features, day_arrays, resolve_target, _runs,
)
import pandas as pd  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LGBM = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=200, colsample_bytree=0.8,
            verbose=-1, n_jobs=1, deterministic=True, force_row_wise=True)
ENS_SEEDS = list(range(10))
AUG = BASE_FEATS + SWEEP_FEATS
N_FLOOR = 500          # capacity-floor permutations (GPT major: 200->500 for a stabler 97.5pct)
N_SHUF = 1000
B_BOOT = 2000
OOS_END = 20260609
VERSIONS = {"lightgbm": lgb.__version__, "python": sys.version.split()[0]}


def ric(p, y) -> float:
    return float(spearmanr(p, y).statistic)


def ens(tr, cols, ev, seeds=ENS_SEEDS) -> np.ndarray:
    Xtr = tr[cols].to_numpy("f8"); ytr = tr["label"].to_numpy(); Xev = ev[cols].to_numpy("f8")
    return np.mean([lgb.LGBMClassifier(**LGBM, random_state=s).fit(Xtr, ytr).predict_proba(Xev)[:, 1]
                    for s in seeds], axis=0)


def collect_window(dates: list[int]) -> pd.DataFrame:
    bars = bars_by_date(min(dates), max(dates))
    rows = []
    for i, d in enumerate(dates):
        if d not in bars:
            continue
        tr = read_day_trades(d)
        if tr.empty:
            continue
        a = day_arrays(tr); starts, ends = _runs(a["sign"], a["price"], a["ts"])
        for s, e in zip(starts, ends):
            sg = a["sign"][s]; p0 = a["price"][s]; p1 = a["price"][e]
            if (p1 - p0) * sg < REC_TICKS * TICK or a["size"][s:e + 1].sum() < REC_VOL:
                continue
            if int(a["minute"][e]) + RECLAIM_MIN > RTH_LAST:
                continue
            label, _ = resolve_target(a, e, p1, sg)
            if label is None:
                continue
            f = build_features(a, s, e)
            rows.append({**{k: f[k] for k in AUG}, "et_date": d, "label": label})
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(dates)} days, events={len(rows)}")
    df = pd.DataFrame(rows)
    return df[np.isfinite(df[AUG].to_numpy("f8")).all(1)].reset_index(drop=True)


def evaluate(tr: pd.DataFrame, ev: pd.DataFrame, with_ablation: bool = True) -> dict:
    y = ev["label"].to_numpy(); ed = ev["et_date"].to_numpy(); days = np.unique(ed)
    by = {d: np.flatnonzero(ed == d) for d in days}
    pb = ens(tr, BASE_FEATS, ev); pa = ens(tr, AUG, ev)
    base_ric = ric(pb, y); aug_ric = ric(pa, y); delta = aug_ric - base_ric
    rng = np.random.default_rng(SEED)
    boot = np.array([(lambda ix: ric(pa[ix], y[ix]) - ric(pb[ix], y[ix]))(
        np.concatenate([by[d] for d in rng.choice(days, len(days), True)])) for _ in range(B_BOOT)])
    # within-day label shuffle null (fixed predictions)
    shuf = np.empty(N_SHUF)
    for i in range(N_SHUF):
        yp = y.copy()
        for d in days:
            ix = by[d]; yp[ix] = y[ix][rng.permutation(len(ix))]
        shuf[i] = ric(pa, yp) - ric(pb, yp)
    # capacity floor (refit permuted-sweep, single seed = conservative for the ensemble)
    floor = np.empty(N_FLOOR); ytr = tr["label"].to_numpy()
    Xb_tr = tr[BASE_FEATS].to_numpy("f8"); Xb_ev = ev[BASE_FEATS].to_numpy("f8")
    pb1 = lgb.LGBMClassifier(**LGBM, random_state=SEED).fit(Xb_tr, ytr).predict_proba(Xb_ev)[:, 1]
    base1 = ric(pb1, y)
    for i in range(N_FLOOR):
        tr2 = tr.copy(); ev2 = ev.copy()
        tr2[SWEEP_FEATS] = tr[SWEEP_FEATS].to_numpy()[rng.permutation(len(tr))]
        ev2[SWEEP_FEATS] = ev[SWEEP_FEATS].to_numpy()[rng.permutation(len(ev))]
        pr = lgb.LGBMClassifier(**LGBM, random_state=SEED).fit(tr2[AUG].to_numpy("f8"), ytr).predict_proba(
            ev2[AUG].to_numpy("f8"))[:, 1]
        floor[i] = ric(pr, y) - base1
    # ablation (DESIGN-ONLY — FORBIDDEN on the OOS firewall: per-feature OOS results leak which feature worked
    # on the sealed window -> future researcher DoF. with_ablation=False on the OOS read.)
    ablation = ({f: ric(ens(tr, [c for c in AUG if c != f], ev), y) - base_ric for f in SWEEP_FEATS}
                if with_ablation else {})
    # day-balanced + halves
    perday = [ric(pa[by[d]], y[by[d]]) - ric(pb[by[d]], y[by[d]])
              for d in days if len(by[d]) >= 10 and len(np.unique(y[by[d]])) == 2]
    h = len(days) // 2
    half = [ric(pa[ix], y[ix]) - ric(pb[ix], y[ix]) for ix in
            (np.concatenate([by[d] for d in days[:h]]), np.concatenate([by[d] for d in days[h:]]))]
    return {
        "base_ric": base_ric, "aug_ric": aug_ric,
        "delta": delta, "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
        "mde95": float(1.96 * boot.std()), "n": int(len(ev)), "days": int(len(days)),
        "capacity_floor_p975": float(np.percentile(floor, 97.5)), "capacity_floor_mean": float(floor.mean()),
        "within_day_shuffle_p975": float(np.percentile(shuf, 97.5)),
        "day_balanced": float(np.mean(perday)) if perday else float("nan"),
        "halves": [float(half[0]), float(half[1])],
        "ablation": {k: float(v) for k, v in ablation.items()},
        "class_balance": float(y.mean()),
    }


def _verify_hash(par: dict) -> None:
    """GPT M1: refuse the OOS read if params.json was edited after finalize (hash IS the pre-registration)."""
    d = {k: v for k, v in par.items() if k != "sha256"}
    h = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if h != par.get("sha256"):
        raise SystemExit(f"PARAMS HASH MISMATCH — refusing OOS read. stored {par.get('sha256')} != computed {h}")


def verdict(r: dict, mde: float, cov_floor_days: int) -> str:
    if r["days"] < cov_floor_days:                                     # GPT M3: underpowered by coverage
        return "INCONCLUSIVE"
    if (r["aug_ric"] > 0 and r["ci_lo"] > 0 and r["delta"] >= mde      # aug must be a real forecast (GPT B2)
            and r["delta"] > r["capacity_floor_p975"] and r["delta"] > r["within_day_shuffle_p975"]
            and (r["day_balanced"] > 0) and min(r["halves"]) >= 0):
        return "PULSE"
    if r["delta"] > 0 and (r["ci_lo"] <= 0 or r["delta"] < mde):
        return "INCONCLUSIVE"
    return "NULL"


def _spec(design_feas: dict, mde_frozen: float, cov_floor_days: int, design_eval_days: int) -> dict:
    return {
        "version": "v3.1", "lightgbm_versions": VERSIONS,
        "event": {"min_ticks": REC_TICKS, "min_vol": REC_VOL, "gap_ms": GAP_MS, "tick": TICK,
                  "decision_time": "sweep last-trade ts", "open_window_min": OPEN_MIN},
        "target": {"geometry": "SYMMETRIC ±K from extreme p1, K=8 ticks (size-decoupled)",
                   "up_sweep": "continuation = p1 + 8t ; reversal = p1 - 8t",
                   "down_sweep": "continuation = p1 - 8t ; reversal = p1 + 8t",
                   "cap_min": RECLAIM_MIN, "resolution": "trade-level first-touch", "tie": "reversal (stop-wins-ties)"},
        "decision_boundary": "RECORD-INDEX safe: t* = final sweep-trade record (positional index e in the "
                             "(ts_ns, sequence) stable-sorted stream). Features use records <= e; resolver uses "
                             "records > e. Same-timestamp post-t* prints are correctly post-decision by sequence.",
        "features": {"baseline": BASE_FEATS, "sweep": SWEEP_FEATS, "interactions": "none (tree learns them)"},
        "model": {"primary": "deterministic 10-seed LightGBM ENSEMBLE (mean predicted prob); M_base=ens(baseline), "
                             "M_aug=ens(baseline+sweep), fit on DESIGN, scored on OOS",
                  "lgbm": LGBM, "ensemble_seeds": ENS_SEEDS, "secondary_nondecisional": "logistic v2.1"},
        "metric": "tie-averaged spearman rank-IC (scipy.stats.spearmanr.statistic) END-TO-END",
        "primary_statistic": "Δ = rankIC(p_aug,y) - rankIC(p_base,y), paired day-block bootstrap",
        "mde_frozen": mde_frozen,
        "design_eval_days": design_eval_days, "oos_days_est": OOS_DAYS_EST, "coverage_floor_days": cov_floor_days,
        "mde_frozen_note": f"GPT M2/M3: mde_frozen = design-eval MDE95 ({design_feas['mde95']:.4f}) × "
                           f"√(design_eval_days {design_eval_days} / oos_days_est {OOS_DAYS_EST}) — OOS-day-scaled, "
                           "FROZEN pre-OOS. At read time it is WIDENED further by √(oos_days_est/actual_oos_days) "
                           f"if OOS has fewer days. coverage_floor_days={cov_floor_days} → INCONCLUSIVE-by-power below it.",
        "split": {"design": [DESIGN_START, DESIGN_END], "oos": [OOS_START, OOS_END], "oos_sealed": True},
        "nulls": {"capacity_floor": f"permuted-sweep refit tree, N={N_FLOOR} (decisive DoF control)",
                  "within_day_label_shuffle": f"N={N_SHUF} (preserves daily base-rate + clustering)",
                  "dropped_vs_protocol5": "forward-shift subsumed by structural leak tests (5/5 green); "
                                          "matched-donor-day-swap impractical at event aggregation"},
        "verdict_bands": "INCONCLUSIVE if oos_days < coverage_floor_days (underpowered). Else PULSE = aug_rankIC>0 "
                         "AND ci_lo>0 AND Δ>=mde_effective AND Δ>capacity_floor_p975 AND Δ>within_day_shuffle_p975 "
                         "AND day_balanced>0 AND both halves>=0; INCONCLUSIVE = Δ>0 but ci_lo<=0 or Δ<mde_effective; "
                         "else NULL. A NULL kills MBP-1 sweep-geometry-over-OFI, NOT MBO Layer-2 absorption.",
        "output_firewall": "the OOS read emits ONLY: base_rankIC, aug_rankIC, Δ rank-IC, day-block CI, capacity "
                           "floor, within-day-shuffle null, day-balanced Δ, both chronological halves Δ, "
                           "coverage/drop counts, verdict. PER-FEATURE ABLATION IS DESIGN-ONLY (forbidden on OOS — "
                           "it would leak which feature worked on the sealed window). No sub-bucket / other-horizon / post-hoc change.",
        "design_feasibility": design_feas,
        "provenance": {"hardening": "f25dc3b", "review": "3aef854"},
        "NOTE": "v3.1 CANDIDATE for lock review. NOT locked. design_feasibility is DESIGN out-of-fold (NOT the OOS verdict).",
    }


def run_design(quick: int | None = None) -> None:
    dates = list_design_dates(DESIGN_START, DESIGN_END)
    if quick:
        dates = dates[:quick]
    df = collect_window(dates)
    days = np.sort(df["et_date"].unique()); cut = days[len(days) // 2]
    tr = df[df["et_date"] < cut]; ev = df[df["et_date"] >= cut]
    r = evaluate(tr, ev)
    mde_frozen = round(r["mde95"] * (r["days"] / OOS_DAYS_EST) ** 0.5, 4)   # GPT M2: OOS-day-scale the design-eval MDE
    cov_floor = round(0.7 * OOS_DAYS_EST)                                    # GPT M3: coverage floor (days)
    par = _spec(r, mde_frozen, cov_floor, r["days"])
    blob = json.dumps(par, sort_keys=True, separators=(",", ":")).encode()
    par["sha256"] = hashlib.sha256(blob).hexdigest()
    (OUT_DIR / "params_candidate_v3.json").write_text(json.dumps(par, indent=2), encoding="utf-8")
    L = ["# v3.1 LOCKED-READ — DESIGN finalize (DESIGN out-of-fold; OOS NOT read)\n",
         f"- model: deterministic 10-seed LightGBM ensemble | metric: tie-averaged spearman | lightgbm {VERSIONS['lightgbm']}",
         f"- eval n {r['n']:,} over {r['days']} days | class balance {r['class_balance']*100:.1f}%",
         f"- **Δ rank-IC {r['delta']:+.4f}** | day-block CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | MDE95 {r['mde95']:.4f}",
         f"- capacity floor 97.5pct {r['capacity_floor_p975']:+.4f} (mean {r['capacity_floor_mean']:+.4f}) | "
         f"within-day-shuffle 97.5pct {r['within_day_shuffle_p975']:+.4f}",
         f"- day-balanced {r['day_balanced']:+.4f} | halves {r['halves'][0]:+.4f} / {r['halves'][1]:+.4f}",
         "- ablation (carried = Δ − drop): " + ", ".join(f"{k} {r['delta']-v:+.4f}" for k, v in r["ablation"].items()),
         f"- would-be DESIGN-OOF verdict (NOT the OOS verdict): **{verdict(r, mde_frozen, cov_floor)}**",
         f"\nparams_candidate_v3.json sha256 `{par['sha256']}`  (mde_frozen={mde_frozen}, "
         f"design-eval MDE95 {r['mde95']:.4f}, cov_floor {cov_floor}d)",
         "\n_For lock review. NOT locked. OOS never read. The OOS verdict comes only from `--oos --confirm-locked`._"]
    (OUT_DIR / "OOS_READ_DESIGN.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def list_oos_dates() -> list[int]:
    out = []
    for part in sorted(MBP1_NQ.glob("date=*")):
        try:
            d = int(part.name.removeprefix("date=").replace("-", ""))
        except ValueError:
            continue
        if OOS_START <= d <= OOS_END and (part / "part-000.parquet").exists():
            out.append(d)
    return out


def run_oos() -> None:
    pf = OUT_DIR / "params_candidate_v3.json"
    par = json.loads(pf.read_text(encoding="utf-8"))
    _verify_hash(par)                                                 # GPT M1: refuse if params tampered
    print(f"LOCKED OOS READ — params sha256 {par['sha256']} (verified), mde_frozen {par['mde_frozen']}")
    tr = collect_window(list_design_dates(DESIGN_START, DESIGN_END))   # fit on FULL DESIGN
    ev = collect_window(list_oos_dates())                              # the sealed OOS — read ONCE
    r = evaluate(tr, ev, with_ablation=False)                          # NO per-feature ablation on OOS (firewall)
    est = par["oos_days_est"]
    mde_eff = round(par["mde_frozen"] * (est / min(r["days"], est)) ** 0.5, 4)   # GPT M2: widen if OOS has fewer days
    v = verdict(r, mde_eff, par["coverage_floor_days"])
    L = [f"# v3.1 OOS VERDICT = {v}\n",                                # firewall: ONLY these fields emitted on OOS
         f"- base_rankIC {r['base_ric']:+.4f} | aug_rankIC {r['aug_ric']:+.4f} | **Δ rank-IC {r['delta']:+.4f}**",
         f"- day-block CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | MDE_frozen {par['mde_frozen']} | "
         f"MDE_effective {mde_eff} | capacity-floor 97.5pct {r['capacity_floor_p975']:+.4f} | "
         f"within-day-shuffle 97.5pct {r['within_day_shuffle_p975']:+.4f}",
         f"- day-balanced Δ {r['day_balanced']:+.4f} | halves {r['halves'][0]:+.4f}/{r['halves'][1]:+.4f} | "
         f"coverage n {r['n']:,} over {r['days']}d (floor {par['coverage_floor_days']}d)"]
    (OUT_DIR / "OOS_VERDICT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--design", action="store_true")
    p.add_argument("--oos", action="store_true")
    p.add_argument("--confirm-locked", action="store_true")
    p.add_argument("--quick", type=int, default=None, help="DESIGN-mode smoke: first N dates")
    a = p.parse_args(argv)
    if a.oos:
        if not a.confirm_locked:
            print("REFUSED: --oos requires --confirm-locked (the sealed OOS is spent ONCE, only after Ben locks).")
            return 2
        run_oos()
    else:
        run_design(a.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
