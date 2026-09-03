# Quant Research Samples

Three condensed examples of my independent quantitative research: prediction markets, CME order flow, and relative-value futures.

The pattern in each is the same. Take an idea from raw data, pre-register the test where it matters, evaluate out of sample, then check whether it survives realistic execution. Negative results are kept as written. Two of the three below end in a rejection.

| Sample | Question | What happened |
|---|---|---|
| [nq-orderflow](nq-orderflow/) | Do sweep/absorption features predict NQ continuation vs. reversal beyond an order-flow baseline? | Yes, out of sample: Δ rank-IC +0.030 over 66,341 events on a sealed window, one pre-registered read. No as a trade: −1.7 ticks per trade after honest fills. Rejected. |
| [prediction-markets](prediction-markets/) | Can a model beat the NFL closing line? Where is Kalshi's sports pricing wrong? | Market baseline over 4,349 games. Blending in a competent Elo makes the forecast worse. Ladder arbitrage: zero violations. Maker edge on props: positive but not conclusive. |
| [relative-value-futures](relative-value-futures/) | Is there a systematic futures pairs edge that survives out of sample? | Selecting pairs by in-sample Sharpe does not generalize (corr +0.12). Economically grounded cointegrated pairs do: a 12-pair book, OOS Sharpe +1.44, full-sample max drawdown −14%, positive in every OOS year and across the whole parameter grid. |

Each folder has a README with the method and results, a result figure, the verdict documents where they exist, and the code that produced the numbers. The prediction-markets baseline runs from a clone with no credentials.

Most of my research and data infrastructure is private: a CME MBO/MBP-1 warehouse, a futures research workstation, and a pre-registration-based research lab. These are excerpts. Implementation is AI-assisted; the hypotheses, data rules, validation standards, and verdicts are mine.
