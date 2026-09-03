"""Leak tests for nq_sweep_absorption_v0 (PROTOCOL.md lock checklist).

Proves the no-lookahead contract is STRUCTURAL: build_features reads only ts <= t*; resolve_target reads only
ts > t*. Runs on one real DESIGN day (OOS never touched).
  & backend/.venv/Scripts/python.exe experiments/nq_sweep_absorption_v0/tests/test_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from experiments.nq_sweep_absorption_v0.design_diagnostics import (  # noqa: E402
    BASE_FEATS, REC_TICKS, REC_VOL, RECLAIM_MIN, RTH_LAST, SWEEP_FEATS, TICK,
    _runs, build_features, day_arrays, read_day_trades, resolve_target,
)

TEST_DAY = 20250501
_KEYS = ("price", "ts", "size", "minute", "sign", "bid_sz", "ask_sz", "bid_px", "ask_px",
         "signed_vol", "prior_high", "prior_low")


def _qualifying(a: dict, limit: int = 50) -> list[tuple]:
    starts, ends = _runs(a["sign"], a["price"], a["ts"])
    out = []
    for s, e in zip(starts, ends):
        sg = a["sign"][s]; p1 = a["price"][e]
        if (p1 - a["price"][s]) * sg < REC_TICKS * TICK or a["size"][s:e + 1].sum() < REC_VOL:
            continue
        if int(a["minute"][e]) + RECLAIM_MIN > RTH_LAST:
            continue
        out.append((int(s), int(e), float(p1), float(sg)))
        if len(out) >= limit:
            break
    return out


_A = day_arrays(read_day_trades(TEST_DAY))
_RUNS = _qualifying(_A)
assert _RUNS, "no qualifying runs on the test day"


def _trunc(a: dict, e: int) -> dict:
    return {k: a[k][:e + 1].copy() for k in _KEYS}


def _shock_after(a: dict, e: int) -> dict:
    b = {k: a[k].copy() for k in _KEYS}
    n = len(b["ts"])
    if e + 1 < n:
        b["ts"][e + 1:] = b["ts"][e] + np.arange(1, n - e) * 10**12   # keep ts SORTED but garbage values
        for k in [k for k in _KEYS if k != "ts"]:
            b[k][e + 1:] = -999_999.0
    return b


def _shock_before(a: dict, e: int) -> dict:
    b = {k: a[k].copy() for k in _KEYS}
    if e > 0:
        for k in [k for k in _KEYS if k != "ts"]:
            b[k][:e] = -999_999.0      # ts left intact so it stays sorted for searchsorted in resolve_target
    return b


def _feat_eq(d1: dict, d2: dict) -> bool:
    return d1.keys() == d2.keys() and all(d1[k] == d2[k] for k in d1)


def test_feature_keys_complete():
    s, e, _, _ = _RUNS[0]
    feat = build_features(_A, s, e)
    missing = set(BASE_FEATS + SWEEP_FEATS) - set(feat)
    assert not missing, f"missing features: {missing}"


def test_truncate_after_tstar_reproduces_features():
    for s, e, _, _ in _RUNS:
        assert _feat_eq(build_features(_A, s, e), build_features(_trunc(_A, e), s, e)), f"trunc mismatch @e={e}"


def test_shock_after_tstar_features_invariant():
    for s, e, _, _ in _RUNS:
        assert _feat_eq(build_features(_A, s, e), build_features(_shock_after(_A, e), s, e)), f"shock-after @e={e}"


def test_target_invariant_to_pre_tstar_shock():
    for s, e, p1, sg in _RUNS:
        assert resolve_target(_A, e, p1, sg) == resolve_target(_shock_before(_A, e), e, p1, sg), f"pre-shock @e={e}"


def test_target_does_read_post_tstar():
    changed = sum(resolve_target(_A, e, p1, sg) != resolve_target(_shock_after(_A, e), e, p1, sg)
                  for _, e, p1, sg in _RUNS)
    assert changed > 0, "resolve_target ignored post-t* data — it must read the future to resolve the barrier"


def test_same_timestamp_after_tstar_no_feature_leak():
    """GPT Blocker-3: a record sharing t*'s timestamp but AFTER it (index e+1) must NOT enter features, and
    the resolver MUST treat it as post-t* (record-index boundary, not timestamp-only)."""
    done = 0
    for s, e, p1, sg in _RUNS:
        if e + 1 >= len(_A["ts"]):
            continue
        b = {k: _A[k].copy() for k in _KEYS}
        b["ts"][e + 1] = b["ts"][e]                       # same timestamp as t*, but index e+1 (after, by sequence)
        b["price"][e + 1] = p1 + (9999 if sg > 0 else -9999)   # extreme: would touch a barrier if it leaked
        b["signed_vol"][e + 1] = 1e9
        assert _feat_eq(build_features(_A, s, e), build_features(b, s, e)), f"same-ts feature leak @e={e}"
        assert resolve_target(b, e, p1, sg)[0] is not None, "resolver must see the same-ts post-t* record"
        done += 1
        if done >= 5:
            break
    assert done > 0, "no run with a following record to test"


def main() -> int:
    tests = [test_feature_keys_complete, test_truncate_after_tstar_reproduces_features,
             test_shock_after_tstar_features_invariant, test_target_invariant_to_pre_tstar_shock,
             test_target_does_read_post_tstar, test_same_timestamp_after_tstar_no_feature_leak]
    fails = 0
    print(f"leak tests on DESIGN day {TEST_DAY} ({len(_RUNS)} qualifying runs):")
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            fails += 1; print(f"  FAIL  {t.__name__}: {exc}")
    print("ALL GREEN" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
