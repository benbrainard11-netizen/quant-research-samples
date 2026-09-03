"""Frozen constants for gamma_0dte_v0. Never inline these numbers elsewhere
(house law: named constants only). Anything sensitivity-tested later is flagged."""
from pathlib import Path

# --- data locations ---
WALLS_SPX = Path(r"D:\data\processed\walls\walls_v2.parquet")
PANEL_ROOT = Path(r"D:\data\processed\option_panels\panel\root=SPXW")
ES_BARS = Path(r"D:\data\processed\bars\timeframe=1m\symbol=ES.c.0")
OUT = Path(__file__).with_name("out")

# --- contract economics ---
SPX_MULTIPLIER = 100.0          # $ per index point per contract
COMMISSION_PER_LEG = 1.00       # $ per option leg per fill (frozen; sensitivity-tested)

# --- session clock, ET milliseconds-of-day ---
TZ = "America/New_York"
RTH_OPEN_MS = 34200000          # 09:30 ET
RTH_CLOSE_MS = 57600000         # 16:00 ET (SPXW PM settlement)

# --- H1 PIN leg (prior-day long gamma) ---
H1_ENTRY_MS = 37800000          # 10:30 ET
H1_SHORT_DELTA = 0.15           # target |delta| of each short strike
H1_WING_PTS = 25.0              # long wing offset beyond each short (SPX pts)

# --- H2 TREND leg (prior-day short gamma) ---
H2_ARM_MS = 36000000            # 10:00 ET (earliest flip-cross trigger)
H2_LONG_DELTA = 0.30            # target |delta| of the bought option
H2_STOP_PREMIUM_FRAC = 0.50     # exit if premium falls to 50% of entry cost
H2_TIME_STOP_MS = 55800000      # 15:30 ET forced exit

# --- H3 full stack: ES orderflow trigger (MBP-1, 2025-05+) ---
MBP1_ES = Path(r"D:\data\raw\databento\mbp-1\symbol=ES.c.0")
OF_WINDOW_MIN = 5              # trailing window for the net-delta imbalance ratio
OF_THRESHOLD = 0.15            # |buy-sell|/(buy+sell) burst level (frozen on signal scale:
                               # 5-min daily-max ~0.20, so this fires on ~97% of days)
OF_ARM_MS = 36000000           # 10:00 ET (earliest orderflow trigger)

# --- evaluation discipline ---
HOLDOUT_YEAR = 2026             # sealed; never trained on (house Law: 2026 sealed)
TRAIN_YEARS = (2024, 2025)      # clean panels restored 2026-07-21; orderflow needs 2025-05+
