"""nflverse data loaders with local caching.

Deliberately no nfl_data_py dependency: the release files are plain
CSV/parquet over HTTPS, and pulling them directly means one less package to
break on a pandas upgrade.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
PARTICIPATION_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
SNAPS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet"
INJURIES_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"


def _download(url: str, dest: pathlib.Path, refresh: bool = False) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not refresh:
        return dest
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_games(refresh: bool = False) -> pd.DataFrame:
    """Schedule + closing market lines, 1999-present.

    Key columns: spread_line (positive = home favored), total_line,
    home/away_moneyline, home/away_spread_odds, result (home margin),
    plus roof, surface, temp, wind, qb names, coaches, rest days.
    """
    path = _download(GAMES_URL, RAW / "games.csv", refresh)
    return pd.read_csv(path, low_memory=False)


def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Play-by-play. ~50k rows and ~380 columns per season."""
    frames = []
    for s in seasons:
        path = _download(PBP_URL.format(season=s), RAW / f"pbp_{s}.parquet", refresh)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_participation(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Personnel, formation, defenders in box. 2016-2023 only."""
    frames = []
    for s in seasons:
        path = _download(
            PARTICIPATION_URL.format(season=s), RAW / f"participation_{s}.parquet", refresh
        )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_snap_counts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Per-player offensive/defensive/ST snap counts and shares, 2012-present."""
    frames = []
    for s in seasons:
        path = _download(SNAPS_URL.format(season=s), RAW / f"snaps_{s}.parquet", refresh)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_injuries(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly injury reports with practice status and game designation."""
    frames = []
    for s in seasons:
        path = _download(
            INJURIES_URL.format(season=s), RAW / f"injuries_{s}.parquet", refresh
        )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


# These releases are single combined files covering all seasons, not per-season.
COMBINED_URL = "https://github.com/nflverse/nflverse-data/releases/download/{rel}/{name}.parquet"


def _load_combined(rel: str, name: str, refresh: bool = False) -> pd.DataFrame:
    path = _download(COMBINED_URL.format(rel=rel, name=name), RAW / f"{name}.parquet", refresh)
    return pd.read_parquet(path)


def load_nextgen(kind: str = "passing", refresh: bool = False) -> pd.DataFrame:
    """Next Gen Stats: tracking-derived team and player aggregates.

    kind is one of passing, rushing, receiving. These are derived from the
    same tracking system behind expected rushing yards.
    """
    return _load_combined("nextgen_stats", f"ngs_{kind}", refresh)


def load_ftn_charting(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Manually charted play detail: motion, play action, blitz, screens."""
    frames = []
    for s in seasons:
        path = _download(
            f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{s}.parquet",
            RAW / f"ftn_charting_{s}.parquet", refresh)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_depth_charts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly depth charts - the cleanest available proxy for declared role."""
    frames = []
    for s in seasons:
        path = _download(
            f"https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{s}.parquet",
            RAW / f"depth_charts_{s}.parquet", refresh)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def completed_games_with_market(games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regular + postseason games that have a final score and both moneylines."""
    g = load_games() if games is None else games
    g = g[g["result"].notna()]
    g = g[g["home_moneyline"].notna() & g["away_moneyline"].notna()]
    g = g[g["spread_line"].notna()]
    g = g.copy()
    g["home_win"] = (g["result"] > 0).astype(int)
    # Ties are rare (~0.1%) but they are not home wins and they break binary
    # scoring, so drop them rather than silently mislabel.
    g = g[g["result"] != 0]
    return g.sort_values(["season", "week"]).reset_index(drop=True)
