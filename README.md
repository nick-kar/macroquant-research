# MacroQuant — Point-in-Time Market Risk Research

**Research question:** Do economic data vintages improve five-session volatility forecasts and risk-controlled portfolio decisions beyond price-history baselines?

A reproducible quantitative research project using SPY, TLT and GLD market history and dated ALFRED economic snapshots. It includes statistical/ML model comparison, chronological validation, label-purged walk-forward refits, paired moving-block bootstrap inference and a turnover/cost-aware portfolio simulator.

Start with **START_HERE_WINDOWS.md**. Read the executed findings in **reference_results/report.html** before adding a claim to your CV. This is a financial research project, not another general AI dashboard.

## Implemented research

| Component | Implementation |
| --- | --- |
| Real inputs | Historical Yahoo OHLC/adjusted closes and public ALFRED vintage CSV downloads |
| Universe | Predefined three-ETF research universe; all had begun trading before the analysis period |
| Macroeconomic features | Vintage-specific unemployment level/change, CPI year-on-year inflation, policy rate |
| Price features | Backward-looking volatility, momentum and downside measures |
| Models | RV20, EWMA, GARCH(1,1), Ridge price/macro, histogram gradient boosting price/macro |
| Validation | 2014–2018 parameter selection and champion selection |
| Test | 2019–2025 prequential forecasts, periodic refits using only fully available labels |
| Statistical assessment | Date-paired cross-asset loss differences, circular block bootstrap, two-comparison Holm adjustment |
| Portfolio simulation | Next-open entry, five-session holding intervals, daily weight drift, cash, caps, no leverage, entry/rebalance/liquidation costs |
| Robustness | Model results by year; 0/5/10/25 bps one-way cost sensitivity; full validation search log |
| Audit | Configuration hash, source hashes, macro vintages, training label-end audits, forecasts and weights |
| Forward extension | Separate append-only-through-CLI dated forecast ledger; no broker execution |

## Reproduce

Python 3.12:
```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m macroquant run --output results/my_run
```
Windows users should use the explicit virtual-environment Python path in START_HERE_WINDOWS.md. Network sources are cached under data/cache. The first run can take several minutes; later runs reuse the cache.

## Why this is quantitative research

Model choice is evaluated against strong baselines and realistic information timing. There is no guaranteed profitable strategy, fake adoption metric or claimed commercial track record. A finding that macro inputs add no useful forecasting improvement is a legitimate result. The full report includes negative outcomes and cost sensitivity rather than presenting only the highest historical Sharpe ratio.

## Material limits

The study is retrospective, not prospectively registered. The three surviving ETFs were chosen today, so the universe remains subject to selection bias; it does not represent all historical investable funds. Yahoo adjusted historical data are a research return proxy and can be corrected by the provider. This is not an institutional point-in-time market-data feed. Macroeconomic features use actual requested historical vintages with conservative date-level availability; no intraday release timing claim is made.

The allocation simulation uses daily adjusted-open ratios, zero cash interest, a proportional cost approximation and no market-impact model. Drawdowns omit within-session extremes. It does not establish live fill quality, capacity or future trading profitability. GARCH convergence failures, if any, are logged and fall back to EWMA. The final test is not used for hyperparameter selection, but the protocol itself was devised retrospectively and the report must not be called a prospective trading track record.

## Repository contents

- macroquant/ — data, features, models, inference, backtesting, reports and CLI
- config/research.json — fixed research specification
- tests/ — timing, leakage, costs, constraints and inference tests
- reference_results/ — executed research outputs and figures
- docs/RESEARCH_PROTOCOL.md — exact information and execution timing
- docs/DATA_AND_PROVENANCE.md — endpoint contracts and data limitations
- docs/CV_AND_INTERVIEW.md — accurate CV language and interview preparation
- docs/VERIFICATION.md — executed checks and limitations
- .github/workflows/ — offline software checks and optional full research rerun

## Sources

- Yahoo chart downloads: source URLs and retrieval hashes are recorded in provenance.json. Review Yahoo's data-use terms before any redistribution or use beyond personal research.
- https://alfred.stlouisfed.org/help/downloaddata
- https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html

No raw market price files are redistributed in this repository. Data downloads can fail; there is no synthetic fallback for research execution.
