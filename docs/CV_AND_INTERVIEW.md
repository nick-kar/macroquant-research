# CV and interview use

## CV entry after your local acceptance run

**MacroQuant — Quantitative Market Risk Research:** Compared seven statistical and ML volatility-forecasting pipelines across three ETFs using historical market data and point-in-time macroeconomic vintages. Implemented label-purged walk-forward validation, turnover- and transaction-cost-aware portfolio backtesting, paired block-bootstrap inference and macro-feature ablation. Python, pandas, scikit-learn, GARCH, econometrics.

This wording focuses on implemented methodology rather than promising returns. Add actual measured findings only if you understand their interpretation. Do not describe historical simulations as a live trading track record, commercial quant employment or profitable investment advice.

## What to show in an interview

1. State the research question and economic motivation.
2. Show why a macro observation's measurement month is different from its release/vintage date.
3. Explain the close-t → next-open entry → five-held-interval target and why the latest incomplete training label is excluded.
4. Show the complete validation search, not only the winning configuration.
5. Compare a simple baseline with a more complex model and explain a negative result if present.
6. Explain why a Sharpe ratio can change after costs, risk targeting or data-timing corrections.
7. Show macro-minus-price loss contrasts with uncertainty; distinguish statistical and economic significance.
8. Explain that ETFs selected today and adjusted historical prices introduce limits even when the feature pipeline avoids explicit future leakage.
9. Open training_audit.csv, source hashes and model forecasts to substantiate the timing claims.
10. If you use the paper extension, show timestamped records and explain that a few forecasts are not a validated trading track record.

## Role alignment

Strongest: financial data science, quantitative analyst/research roles in systematic investing, market-risk modelling and portfolio research. Less direct: high-frequency market making, exchange connectivity and derivatives pricing. A domain project supplements probability, statistics and coding assessment preparation; it does not replace it.

## Describe your contribution honestly

Built using an assisted implementation. Personally reproduce the run, inspect outputs, explain the design and make at least one justified analysis of the results. Do not claim an independently discovered profitable strategy from supplied code. The value is defensible quantitative reasoning and reliable implementation.
