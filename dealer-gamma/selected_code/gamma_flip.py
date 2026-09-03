"""Real intraday dealer-gamma flip from the panel's per-strike gamma x prior-OI.

net_gex(S) = sum over ALL listed contracts of  sign * BS_gamma(S) * oi_prior * S^2
  sign = +1 call, -1 put (dealers long calls / short puts; matches walls_v2's
  net-dealer-gamma convention, verified 2026-07-20: net_gex@spot < 0 on the
  gex_proxy<0 trend days). BS gamma recomputed at each hypothetical spot under
  sticky-strike vols (each contract's IV fixed). The flip = the zero crossing of
  net_gex(S) nearest to spot; on short-gamma days it sits ~0.1-1.0% from spot,
  unlike walls_v2.zero_gamma (~15% below spot, unusable intraday).

Computed once per day at the arm minute: OI is fixed intraday, so the flip is a
stable structural level (recomputing every minute is not worth the cost).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

_MS_PER_YEAR = 365.0 * 24 * 3600 * 1000
FLIP_BAND = 0.05       # search +/-5% of spot
FLIP_NPTS = 61
MIN_CONTRACTS = 50     # below this the profile is too thin to trust


def _prep(full, date_int, ms):
    s = full[(full["ms_of_day"] == int(ms)) & full["implied_vol"].notna() &
             full["oi_prior"].notna()]
    s = s[(s["implied_vol"] > 0) & (s["oi_prior"] > 0)]
    if len(s) < MIN_CONTRACTS:
        return None
    now = pd.to_datetime(str(int(date_int)))
    dte = (pd.to_datetime(s["expiration"].astype(int).astype(str)) - now
           ).dt.days.clip(lower=0).to_numpy()
    tte = dte / 365.0 + max(config.RTH_CLOSE_MS - ms, 0) / _MS_PER_YEAR
    return {
        "K": s["strike"].to_numpy(float), "iv": s["implied_vol"].to_numpy(float),
        "oi": s["oi_prior"].to_numpy(float), "tte": tte,
        "sign": np.where(s["right"].astype(str).to_numpy() == "C", 1.0, -1.0),
    }


def _net_gex_at(S, p):
    tte = p["tte"]
    ok = tte > 0
    d1 = np.where(ok, (np.log(S / p["K"]) + 0.5 * p["iv"] ** 2 * tte) /
                  (p["iv"] * np.sqrt(np.where(ok, tte, 1.0))), 0.0)
    gamma = np.where(ok, norm.pdf(d1) / (S * p["iv"] * np.sqrt(np.where(ok, tte, 1.0))), 0.0)
    return float(np.sum(p["sign"] * gamma * p["oi"] * S * S))


def daily_flip(full, date_int, ms, spot):
    """Zero crossing of net_gex(S) nearest to `spot`, or None if none in band."""
    p = _prep(full, date_int, ms)
    if p is None:
        return None
    grid = np.linspace(spot * (1 - FLIP_BAND), spot * (1 + FLIP_BAND), FLIP_NPTS)
    gex = np.array([_net_gex_at(S, p) for S in grid])
    ch = np.where(np.diff(np.sign(gex)) != 0)[0]
    if len(ch) == 0:
        return None
    flips = [grid[i] + (grid[i + 1] - grid[i]) * (0 - gex[i]) / (gex[i + 1] - gex[i])
             for i in ch]
    return float(min(flips, key=lambda x: abs(x - spot)))
