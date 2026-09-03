# COST-TEST v2 — does PASSIVE entry + move-size selection rescue the edge? (DESIGN-OOF; NOT a verdict)

> Honest maker fills on the real MBP-1 tick path: strict trade-THROUGH (AT-touch ≠ fill), adverse-barrier-before-
> fill (miss = the runner got away), TAKER ±8t exit, MISS-INCLUSIVE accounting (a miss falls back to v1's taker
> P&L). So maker beats v1 only if it wins on the events it fills passively. NQ $5/tick. DEV only — no verdict.

- events 81,020 over 183 days | continuation 35.8% → baseline = always-fade
- **PRIMARY** (fade · through · OFFSET=1 · all · miss-incl · exit 1.5t): net **-1.731 t** ($-8.65/trade) day-block CI [-1.814, -1.645] | fill-rate 64.8% | filled-only -3.019 t
- **the bar — v1 taker fade @2.0t over the same universe:** -2.702 t (CI [-2.779, -2.622])

## Cells (DESIGN-OOF; only the PRIMARY is decisional — the rest are pre-registered sensitivity/secondary)
| cell | net t | CI | fill% | filled-only |
|---|---:|---:|---:|---:|
| PRIMARY_fade_through_o1_all_exit1.5 | -1.731 | [-1.814,-1.645] | 64.8 | -3.019 |
| fade_through_o2_all_exit1.5 | -1.241 | [-1.328,-1.152] | 58.5 | -2.876 |
| fade_q0haircut_o1_all_exit1.5_OPTIMISTIC | -1.674 | [-1.758,-1.587] | 68.6 | -2.554 |
| fade_through_o1_TOPATRtercile_exit1.5 | -1.995 | [-2.153,-1.826] | 61.0 | -3.375 |
| h1directed_through_o1_all_exit1.5_secondary | -1.691 | [-1.774,-1.604] | 64.6 | -3.004 |

## Placebos (guard against a FALSE go)
- random-direction miss-incl net: **-1.041 t** (should be ≈ −friction; >0 ⇒ execution-artifact leak)
- **at-touch − through gap: +0.123 t** (the queue-illusion / fake-edge magnitude)
- filled-only − miss-inclusive: -1.288 t (the adverse-selection magnitude — the events you passively fill are worse fade trades)
- fade fill-rate on its WINS (reversals) 46.5% vs its LOSSES (continuations) 97.5% — the fade edge lives in fast reversals that never return to a passive limit
- shuffled-atr tercile net -1.702 t (≈ all-events ⇒ selection adds nothing)

## Verdict (development economics — NOT a lock)
- **NO, but informative — passive entry IMPROVES on the taker trade by +0.97 t/trade (a fixed +1t entry on 64.8% fills, surviving the honest trade-THROUGH model — NOT eaten by adverse selection), yet the ±8t fade trade stays too unprofitable (-1.73 t net, CI [-1.814, -1.645]) for it to clear 0. The entry leg is ~maxed; futures need a better BASE trade (move-size targets/selection) or options convexity. H1 remains a forecast, not yet a futures trade.**

_DESIGN-OOF / development only. No OOS read, no lock, no live trading. The forward window is the real economic OOS._