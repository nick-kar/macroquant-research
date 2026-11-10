# Research protocol

## Hypotheses

Primary question: do dated macroeconomic features improve forward realized-variance forecasts relative to the same model family using only price history?

Two focal contrasts are specified: Ridge macro minus Ridge price QLIKE, and histogram gradient boosting macro minus its price-only counterpart. Negative differences favour the macro feature set. Validation hyperparameters are selected separately within each feature set; this comparison therefore measures the tuned model pipelines, not a perfectly isolated coefficient experiment.

Secondary question: how do resulting risk-controlled allocations behave under explicit portfolio and cost assumptions? This is an implementation study, not a claim of return-predictive alpha.

## Information and execution clock

For a decision at close t:
1. Price features use adjusted-open returns through open t and adjusted-close returns through close t.
2. Macro features come from the latest month-end vintage conservatively available by t; each snapshot becomes usable on the following calendar day. All transformations, including year-on-year CPI, are computed within that vintage.
3. Orders are assumed filled at the next session's open, t+1.
4. Holdings remain until open t+6, comprising five open-to-open intervals.
5. The target is 252 times the mean squared log adjusted-open return over those five intervals. The zero-conditional-mean approximation is explicit; this is not a de-meaned five-observation sample variance.
6. The label becomes fully known at open t+6. Model training conservatively requires label_end < refit decision date.

Decision positions are spaced five common trading sessions apart, so the previous exit equals the next entry. No overlapping target periods occur in the full aligned test sequence. Asset-specific source calendars are intersected without forward-filling missing prices; discarded row counts are recorded.

## Time partitions and selection

History starts in 2005; features require a 252-session warm-up. Validation decisions occur in 2014–2018. Validation labels crossing the first test decision are purged before selecting hyperparameters. Test decisions occur from 2019 through the last complete holding period in 2025. Ridge alpha candidates: 0.1, 1, 10, 100. Tree leaf-count candidates: 5 and 10; other settings fixed in code.

Each learned model refits every 13 signal periods using all preceding rows whose labels have completed. Standardization and asset encoding are fitted within that eligible training set. During the test, earlier test labels may enter subsequent refits only after becoming available; this is prequential evaluation, not training on future outcomes. Parameter choice and model-family champion are frozen from validation before the test forecasts run. All validation candidates are retained in the search log.

RV20 uses preceding 20 open-return squares. EWMA uses lambda 0.94. GARCH(1,1) uses zero-mean normal returns scaled by 100 and at most 756 preceding observations. Parameters refit every 13 decisions and conditional variance is updated recursively with newly available open returns. The mean forecast over steps 2 through 6 matches the held open-to-open target. Nonconvergence is logged and uses EWMA.

## Forecast and inference metrics

QLIKE = actual/forecast - log(actual/forecast) - 1, with small positive floors. Also report variance MSE and volatility MAE. Loss differences are first averaged across assets at each decision date. Bootstrap samples circular blocks of 13 periods with seed 42 and 1,000 repetitions. Report 95% percentile intervals and centred-bootstrap two-sided p-values; apply Holm adjustment only to the two focal macro contrasts. No claim is made that this corrects an unknown history of strategy searches or regime instability.

## Portfolio model

Forecast portfolios start with inverse-volatility weights, cap each risky weight at 60%, and scale down to a 10% forecast portfolio-volatility target. Correlations use preceding 252 adjusted closes. Leverage and shorting are prohibited; residual cash earns zero.

At rebalance, compare target risky weights with the weights carried from the previous period after price drift. Turnover is the sum of absolute risky-weight changes, not half that sum. Costs are one-way bps times that turnover. The cost approximation removes fees proportionally from NAV before holding-period returns. This is not an order-by-order cash/share fill model. Daily weights drift; no free daily rebalancing is assumed. Initial entry and final liquidation are charged. Cost assumptions: 0, 5, 10, 25 bps; 10 bps is the predeclared main scenario.

Equal-weight periodic rebalancing and buy-and-hold are comparators. They have different risk exposure from volatility-targeted strategies; differences in CAGR cannot be interpreted as pure alpha. Annualized Sharpe uses zero cash return and 252 daily intervals; CAGR uses actual calendar duration. Drawdown is sampled at daily open-based NAV observations, not intraday lows.

## Boundaries

No auto-trader, no real-money orders, no options-pricing claim, no high-frequency market-microstructure claim, and no guarantee of investment performance. The study demonstrates empirical research methodology on real historical data.
