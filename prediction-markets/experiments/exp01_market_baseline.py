"""Experiment 01 - How good is the market, and can a from-scratch model beat it?

This is the first thing to run and the number every later idea gets measured
against. It answers two questions with real data:

  1. How well calibrated is the NFL closing line?
  2. Does a competent independent model (Elo) beat it out of sample?

If the answer to (2) is no - and it will be - that settles the strategy
question. You do not replace the market number. You start from it.

Run:  python experiments/exp01_market_baseline.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sportmodels import evaluate as ev
from sportmodels import market as mk
from sportmodels import nfl_data as nd

FIRST_TEST_SEASON = 2010


def build_elo(
    games: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 55.0,
    revert: float = 0.25,
    scale: float = 400.0,
) -> pd.Series:
    """Walk-forward Elo. Every prediction uses only prior games.

    Ratings revert `revert` of the way to 1500 between seasons to account for
    offseason roster turnover.
    """
    order = games.sort_values(["season", "week", "gameday"]).index
    ratings: dict[str, float] = {}
    last_season: int | None = None
    preds = pd.Series(index=games.index, dtype=float)

    for i in order:
        row = games.loc[i]
        season, home, away = row["season"], row["home_team"], row["away_team"]

        if last_season is not None and season != last_season:
            for t in ratings:
                ratings[t] = 1500.0 + (1.0 - revert) * (ratings[t] - 1500.0)
        last_season = season

        rh = ratings.setdefault(home, 1500.0)
        ra = ratings.setdefault(away, 1500.0)

        p_home = 1.0 / (1.0 + 10.0 ** (-(rh + home_adv - ra) / scale))
        preds.loc[i] = p_home

        # Margin-of-victory multiplier, dampened for the favorite so blowouts
        # by heavy favorites do not inflate ratings without bound.
        margin = abs(row["result"])
        elo_diff = (rh + home_adv - ra) * (1 if row["result"] > 0 else -1)
        mov = np.log(margin + 1.0) * (2.2 / (elo_diff * 0.001 + 2.2))

        actual = 1.0 if row["result"] > 0 else 0.0
        delta = k * mov * (actual - p_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    return preds


def main() -> None:
    games = nd.completed_games_with_market()
    print(f"Loaded {len(games):,} completed games with market prices, "
          f"{games.season.min()}-{games.season.max()}")

    p_home_raw = mk.american_to_prob(games["home_moneyline"].to_numpy())
    p_away_raw = mk.american_to_prob(games["away_moneyline"].to_numpy())
    games["mkt_prop"], _ = mk.devig_proportional(p_home_raw, p_away_raw)
    games["mkt_power"], _ = mk.devig_power(p_home_raw, p_away_raw)
    games["elo"] = build_elo(games)

    vig = mk.overround(p_home_raw, p_away_raw)
    print(f"Mean moneyline overround: {vig.mean():.4f} "
          f"({(vig.mean() - 1) * 100:.2f}% vig)")
    print(f"Break-even at -110: {mk.break_even_prob(-110):.4f}\n")

    # Spread -> probability. sigma is refit each season on prior seasons only.
    spread_pred = pd.Series(index=games.index, dtype=float)
    for season, train, test in ev.season_walk_forward(
        games["season"].to_numpy(), FIRST_TEST_SEASON
    ):
        sigma = mk.fit_margin_sigma(
            games.loc[train, "spread_line"], games.loc[train, "result"]
        )
        spread_pred.loc[test] = mk.spread_to_win_prob(
            games.loc[test, "spread_line"].to_numpy(), sigma
        )
    games["mkt_spread"] = spread_pred

    full_sigma = mk.fit_margin_sigma(games["spread_line"], games["result"])
    print(f"Std dev of (actual margin - spread): {full_sigma:.2f} points\n")

    test = games[games["season"] >= FIRST_TEST_SEASON].copy()
    y = test["home_win"].to_numpy()

    # Home-field-only baseline, also fit walk-forward.
    hfa = pd.Series(index=games.index, dtype=float)
    for season, train, tst in ev.season_walk_forward(
        games["season"].to_numpy(), FIRST_TEST_SEASON
    ):
        hfa.loc[tst] = games.loc[train, "home_win"].mean()
    test["home_only"] = hfa.loc[test.index]

    # Does knowing Elo add anything once you already know the market price?
    blend = np.clip(
        0.75 * test["mkt_power"].to_numpy() + 0.25 * test["elo"].to_numpy(), 1e-6, 1 - 1e-6
    )

    results = [
        ev.summarize("home_field_only", test["home_only"], y),
        ev.summarize("elo_from_scratch", test["elo"], y),
        ev.summarize("market_spread", test["mkt_spread"], y),
        ev.summarize("market_ml_proportional", test["mkt_prop"], y),
        ev.summarize("market_ml_power", test["mkt_power"], y),
        ev.summarize("market_75_elo_25", blend, y),
    ]

    print(f"Out-of-sample: {FIRST_TEST_SEASON}-{int(test.season.max())}, "
          f"n={len(test):,}\n")
    table = ev.compare(results, baseline="market_ml_power")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nMarket calibration (devigged moneyline, home team):")
    print(
        ev.calibration_table(test["mkt_power"], y, bins=10).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    out = pathlib.Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "exp01_baseline.csv", index=False)
    print(f"\nSaved -> {out / 'exp01_baseline.csv'}")


if __name__ == "__main__":
    main()
