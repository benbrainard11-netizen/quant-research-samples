"""Prior-close dealer-gamma regime labels from walls_v2.

Leak rule (PROGRAM law 1): a live 0DTE trade on date T can only know the gamma
structure finalized at the T-1 close. We attach, to each trade date T, the most
recent walls_v2 row STRICTLY BEFORE T (merge_asof, exact matches disallowed).

Sign convention (verified 2026-07-20, n=2,335): gex_proxy > 0 => long-gamma/PIN
(median realized range 0.90%); < 0 => short-gamma/TREND (1.65%).
"""
import pandas as pd

from . import config

PIN = "long_gamma"     # positive GEX: dealers suppress -> price pins
TREND = "short_gamma"  # negative GEX: dealers amplify -> price trends


def load_walls() -> pd.DataFrame:
    w = pd.read_parquet(config.WALLS_SPX)
    w["date"] = w["date"].astype("int64")
    return w.sort_values("date").reset_index(drop=True)


def regime_for_dates(trade_dates) -> pd.DataFrame:
    """One row per trade date T carrying the PRIOR session's gamma structure.

    Columns: date (int T), regime, gex, spx_prevclose, call_wall, put_wall,
    zero_gamma, pin. Rows with no prior walls row are dropped.
    """
    w = load_walls()
    td = pd.DataFrame({"date": sorted(int(d) for d in trade_dates)})
    m = pd.merge_asof(td, w, on="date", direction="backward",
                      allow_exact_matches=False)
    m = m.dropna(subset=["gex_proxy"]).copy()
    m["regime"] = m["gex_proxy"].map(lambda g: PIN if g > 0 else TREND)
    return m.rename(columns={
        "gex_proxy": "gex", "spot": "spx_prevclose",
    })[["date", "regime", "gex", "spx_prevclose",
        "call_wall", "put_wall", "zero_gamma", "pin"]].reset_index(drop=True)
