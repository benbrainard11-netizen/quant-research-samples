# gamma_0dte_v0 — PROGRAM (the constitution; read before adding anything)

A gamma-regime-switched 0DTE SPXW strategy program. The dealer-gamma regime
(from the validated `walls_v2` levels) decides the day's character; the trade
is expressed in 0DTE SPXW options priced from the intraday option panels.
Modeled on the macro_state_v0 / market_state pattern — the program lives,
individual legs die. This document governs every addition.

Direction set 2026-07-20: **regime-switched (both legs)** is the
destination. Build the shared harness once; register the two legs as
**separate hypotheses** so one can pass while the other dies honestly.

---

## The thesis (one paragraph)

Dealer gamma positioning known at the prior close predicts the day's realized
character. **Verified 2026-07-20** on `walls_v2` + ES daily ranges (n=2,335):
prior-day **positive** GEX days realize a median daily range of **0.90%**;
prior-day **negative** GEX days realize **1.65%** (means 1.10% vs 2.01%).
Long gamma suppresses range (dealers sell rallies / buy dips → pin); short
gamma amplifies it (dealers chase → trend). We trade 0DTE into that character.

**Frozen sign convention:** `gex_proxy > 0` = long-gamma / PIN regime;
`gex_proxy < 0` = short-gamma / TREND regime. (Confirmed, not assumed.)

## Layers

**Layer 0 — Shared harness** (durable asset; build first)
- Regime labeller: for trade date T, read the `walls_v2` row for the last
  session ≤ T−1 (prior EOD gamma structure). Emits regime sign + levels
  (`spot, call_wall, put_wall, zero_gamma, pin`). **Leak rule:** today's OI /
  gamma is not final until today's close, so a live 0DTE trade can only know
  T−1's structure. Enforced by construction (shift, never same-day).
- Intraday underlying: SPX has no local intraday cash feed (paywalled). Use
  the **ES future** front-month 1-minute bars, mapped to SPX by
  `SPX_t ≈ SPX_prevclose + (ES_t − ES_prevclose)`. Leak-clean (no panel
  lookahead). Frozen.
- 0DTE option pricer / fill engine: reads the SPXW option panel for date T,
  filters `expiration == T` (0DTE). **Honest fills, no midpoint** (house law +
  0dte-lab rule): BUY at ask, SELL at bid, always. Any mid is
  `_mid_reference_only`. Stop-vs-target resolves conservatively (stop wins on
  ambiguity); every run records `ambiguous_fill_count`.
- Named constants only: SPX multiplier ($100/pt), commission, session hours,
  entry times, wing widths — in one typed config, never inlined.

**Layer 1 — The two registered legs** (each its own kill test)

### H1 — Long-gamma PIN (sell premium)
- **Hypothesis + mechanism:** on prior-day positive-GEX days, SPX pins inside
  the [put_wall, call_wall] channel; realized < implied → a 0DTE defined-risk
  iron condor sold mid-morning expires OTM and collects decay.
- **Frozen entry:** eligible if prior-day `gex_proxy > 0` AND at 10:30 ET the
  SPX proxy is inside [put_wall, call_wall]. Sell 0DTE iron condor: short
  strikes at the nearest listed strike to ~15-delta (from panel greeks) each
  side; long wings +25 SPX pts beyond each short (defined risk).
- **Frozen exit:** hold to settlement; regime-invalidation stop if the SPX
  proxy touches either short strike (buy the tested spread back at ask =
  conservative). Flat by close.
- **Matched control:** the identical condor, same entry clock, on
  prior-day **negative**-GEX days (where the pin shouldn't hold) and on
  regime-agnostic random days. H1 must beat both.
- **Kill / floors:** net expectancy per trade (after honest fills +
  commission) > 0 AND > matched control by t ≥ 2 on TRAIN; 2023+ sign
  stability; wins not concentrated in a single quarter.
- **Unfalsifiability check:** H1 dies if expectancy ≤ 0, ≤ control, or the
  edge is one-quarter-only. A pin that "works because it held" is not scored —
  only the pre-set entry/exit is.

### H2 — Short-gamma TREND (buy directional)
- **Hypothesis + mechanism:** on prior-day negative-GEX days, a break of the
  `zero_gamma` flip is amplified by dealer hedging → a 0DTE long option bought
  on the break rides toward the same-side wall.
- **Frozen entry:** eligible if prior-day `gex_proxy < 0`. Trigger = first
  intraday cross of `zero_gamma` by the SPX proxy after 10:00 ET; direction =
  cross side (up→long, down→short). Buy the nearest-listed ~30-delta 0DTE
  option in the break direction (risk = premium; naturally defined).
  Orderflow confirm (ES MBP-1 net delta in the break direction) is an
  **ablation arm only** — MBP-1 starts 2025-05, so it cannot be a primary
  gate on the 2024 train window.
- **Frozen exit:** target = same-side wall (call_wall long / put_wall short);
  stops = flip re-cross, premium −50%, or 15:30 ET time stop; flat by close.
- **Matched control:** the same long-option purchase at the same trigger on
  prior-day **positive**-GEX days (trend shouldn't pay) and at random intraday
  times. H2 must beat both.
- **Kill / floors:** same bar as H1 (expectancy > 0, > control, t ≥ 2, 2023+
  sign stable, not one-quarter). Convex payoff means low win rate is fine;
  the test is expectancy, not hit rate.
- **Unfalsifiability check:** H2 dies if expectancy ≤ 0, ≤ control, or only a
  handful of trades carry it. A trend that "worked because it trended" is not
  scored — only the pre-set flip-cross entry.

## Laws (apply forever)
1. **No lookahead.** Regime from prior close; underlying from ES (no panel
   parity anchor, which carries a documented whole-day lookahead).
2. **Honest fills only.** Buy at ask, sell at bid. No midpoint fills. Stop
   beats target on ambiguity.
3. **One leg's result never rescues the other.** Two hypotheses, two verdicts.
4. **Dimensionless comparisons across eras** (returns in R / % / delta units,
   never raw premium dollars pooled across vol regimes).
5. **2026 stays sealed.** Train + explore on ≤2025 only; the sealed-2026
   holdout is the sole final arbiter, opened once, ever.
6. **Kill criteria are written before the test runs** (above) and are not
   moved after seeing results. Dead legs go to the journal.

## Data status (verified 2026-07-20)
- Regime labels (`walls_v2`): SPX **2017-01 → 2026-06**, clean. No issue.
- 0DTE option panels (`option_panels/panel/root=SPXW`): **HALF CORRUPTED.**
  259 readable partitions = **all of calendar 2024**; 257 unreadable =
  late-2023 + all of 2025-2026 (interrupted panel rebuild → parquet files with
  no valid footer). Raw 1-min quote source is intact for every year 2020-2026
  (`options_signals_v0/out/intraday_pc/root=SPXW`, 1,157 expirations), so the
  corrupted partitions are **rebuildable locally, no downloads.**
- **Build sequencing forced by the data:**
  - Phase 1 (now): harness + both legs validated on clean **2024** (train).
    Honest caveat — 2024 was a low-vol uptrend year, thin on negative-gamma
    trend days, so H2 gets less training signal than H1.
  - Phase 2: rebuild corrupted SPXW panels (2025-2026 + late-2023) from raw →
    restores more train (2025) + the sealed-2026 holdout.
  - Phase 3: one-shot the sealed-2026 holdout. Final.

## Where things live
- This dir: `config.py` (constants), `regime.py` (labeller), `underlying.py`
  (ES→SPX proxy), `panel_io.py` (0DTE panel reader + fill engine),
  `backtest.py` (the two legs), `controls.py`, `report.py`. One leg = one
  results file under `out/`. Journal: new project line.

## Status
- 2026-07-20: PROGRAM drafted. Sign convention verified. Data corruption
  found + rebuild path confirmed. Signed off: regime-switched (both legs).
- 2026-07-20: **Layer 0 + both legs built and run on clean 2024 (train).**
  Two frozen-definition BUGS caught + fixed pre-verdict (not goalpost moves):
  (1) H2 trigger — walls_v2 `zero_gamma` sits ~15% below spot (deep-OTM
  theoretical crossing), never fires intraday. Replaced with a real per-minute
  net-GEX flip recomputed from panel gamma x prior-OI (`gamma_flip.py`); it
  lands 0.1-1.0% from spot on trend days, verified. (2) H2 exit — a
  flip-recross stop sat at the entry level and strangled 17/18 trades into
  instant scalps; removed.
- 2026-07-20 VERDICT (train, 2024, thin: n=46 H1 / 18 H2): **BOTH LEGS FAIL
  the regime test.** H1 pin condor: +0.048r pin vs +0.041r control (t=0.25) —
  the condor earns 0DTE decay but the regime adds nothing. H2 trend buy:
  -0.126r trend vs +0.488r control (t=-1.45) — the trade is WORSE on trend
  days, inverting the thesis. Consistent with the lab prior
  (`market_state_options_v0`: gamma->vol REAL, gamma->tradeable-edge NULL).
- **Confirmed durable finding:** gex_proxy regime cleanly predicts realized
  RANGE (0.90% pin vs 1.65% trend median, n=2,335) — its real value is
  vol/range CONTEXT + sizing, not a standalone 0DTE entry signal.
- **Residue (train-era only, NOT claims):** (a) non-regime 0DTE premium
  selling has a pulse in calm 2024 (H1 +$114/trade, tail-fragile, unproven out
  of low vol); (b) "buy break -> target wall" pays better in pin/low-vol than
  trend. Both need the 2025 rebuild + sealed-2026 holdout to mean anything.
- **Untested:** the full "regime -> ORDERFLOW -> entry" stack. Only
  regime->entry (no orderflow middle layer) was tested; MBP-1 orderflow starts
  2025-05, so it needs the panel rebuild first.
- 2026-07-21: **Panels REBUILT clean** (see rebuild_spxw_panels.py):
  576 partitions all good — 2024 (259), 2025 May-Dec (174), 2026 Jan-Jun (118).
  gamma vol/range CONTEXT engine shipped (`context.py` -> gamma_context_latest.json).
- 2026-07-21: **H3 full stack built + run on 2025 train** (regime FILTER + ES
  MBP-1 orderflow TRIGGER + 0DTE entry). MBP-1 sign verified (side B=+delta,
  corr +0.65). Orderflow burst trigger = 5-min net-delta imbalance >= 0.15
  (frozen on signal scale, ~97% day coverage).
- 2026-07-21 H3 VERDICT (2025 train, n=31 trend trades): **FAIL floors.**
  mean_r +0.321, +$307/trade, win 45%, but **t_vs_zero=1.18** (not sig);
  regime-blind control +0.063 (t=0.88), pin-day control -0.034 (t=1.21) — point
  estimates lean the thesis (trend > blind > pin) but NOTHING clears t>=2;
  quarter-unstable (Q2 -0.32, Q3 +0.30, Q4 +0.61). Underpowered null.
- **Orderflow pre-check (n=110):** burst direction predicts next-30min direction
  44.5% — mildly CONTRARIAN (flow gets faded, not followed), confirming the lab
  orderflow-null prior. Residue lead (train-era only, NOT a claim): a FADE
  variant (buy opposite the burst) is the direction any edge would live; do not
  test-and-holdout on this same 2025 data (p-hacking).
- **ALL THREE entry hypotheses (H1 pin / H2 trend / H3 full stack) FAIL.** The
  durable deliverable is the gamma vol/range CONTEXT tile. 2026 holdout NOT
  opened — no candidate earned the one shot; it stays sealed for a successor.
