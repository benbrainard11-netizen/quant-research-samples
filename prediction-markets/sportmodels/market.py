"""Odds math.

Everything here is about one question: what probability is the market actually
quoting, and what probability do I need to break even?

Convention used throughout: American odds. Negative = favorite.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def american_to_decimal(odds):
    """American odds -> decimal odds (total return per $1 staked, incl. stake)."""
    odds = np.asarray(odds, dtype=float)
    # np.where evaluates both branches, so guard the denominator to keep the
    # unused branch from warning on odds == 0.
    neg = odds < 0
    return np.where(neg, 1.0 + 100.0 / np.where(neg, -odds, 1.0), 1.0 + odds / 100.0)


def american_to_prob(odds):
    """American odds -> implied probability, vig included.

    This is also the break-even win rate. At -110 it is 0.5238, which is the
    number that kills most bettors: you need to be right 52.38% of the time
    just to stay flat.
    """
    odds = np.asarray(odds, dtype=float)
    # Guard both denominators: np.where evaluates the untaken branch too, and
    # a -100 moneyline would divide by zero in the positive branch.
    neg = odds < 0
    return np.where(
        neg,
        -odds / np.where(neg, -odds + 100.0, 1.0),
        100.0 / np.where(neg, 1.0, odds + 100.0),
    )


# Same computation, different intent. Use this name when you mean "the bar".
break_even_prob = american_to_prob


def overround(p_a, p_b):
    """Total implied probability across both sides. 1.0 = no vig."""
    return np.asarray(p_a, float) + np.asarray(p_b, float)


def devig_proportional(p_a, p_b):
    """Strip vig by scaling both sides to sum to 1.

    Simple and fast, but it is known to shade probability toward the favorite
    relative to better methods. Fine as a default, worth checking against
    `devig_power` when a result looks marginal.
    """
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    s = p_a + p_b
    return p_a / s, p_b / s


def devig_power(p_a, p_b, max_iter: int = 60):
    """Strip vig by finding k such that p_a**k + p_b**k == 1.

    Removes proportionally more vig from the longshot, which is closer to how
    books actually price. Solved by bisection on k in [1, 10].
    """
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)
    lo = np.ones_like(p_a)
    hi = np.full_like(p_a, 10.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # Sum is decreasing in k, so oversized sum means k must go up.
        too_big = (p_a**mid + p_b**mid) > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    k = 0.5 * (lo + hi)
    return p_a**k, p_b**k


def spread_to_win_prob(spread_line, sigma: float = 13.2):
    """Point spread -> win probability, normal approximation.

    `spread_line` follows the nflverse convention: positive means the home team
    is favored by that many points. sigma is the standard deviation of
    (actual margin - spread), historically ~13.
    """
    return norm.cdf(np.asarray(spread_line, float) / sigma)


def fit_margin_sigma(spread_line, actual_margin) -> float:
    """Estimate sigma from data. Fit this on training seasons only."""
    resid = np.asarray(actual_margin, float) - np.asarray(spread_line, float)
    return float(np.std(resid, ddof=1))


def expected_value(p, odds):
    """EV per $1 staked at these odds given true probability p."""
    b = american_to_decimal(odds) - 1.0
    p = np.asarray(p, float)
    return p * b - (1.0 - p)


def kelly_fraction(p, odds, cap: float = 1.0):
    """Full-Kelly stake as a fraction of bankroll. Negative edge -> 0.

    In practice bet a fraction of this (quarter-Kelly is common) because `p`
    is an estimate, not the truth, and Kelly is brutally sensitive to that.
    """
    b = american_to_decimal(odds) - 1.0
    p = np.asarray(p, float)
    f = (p * (b + 1.0) - 1.0) / b
    return np.clip(f, 0.0, cap)
