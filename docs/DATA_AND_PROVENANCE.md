# Data and provenance

## Market data

The code requests historical daily quotes from Yahoo's public chart endpoint and records the exact URL, response SHA-256, retrieval time and row range. Data comprise SPY, TLT and GLD from 2005-01-01 through 2025-12-31. The raw quote open is multiplied by adjusted_close / raw_close to create an adjusted-open return proxy. This does not mean adjusted historical share prices could literally have been traded.

There is no institutional point-in-time price database or independent corporate-action reconciliation in this release. Historical corrections and provider adjustment conventions can affect results. A later redownload may differ; compare response hashes. The implementation does not silently substitute Stooq or generated prices if Yahoo is unavailable.

Raw response caches are excluded from the ZIP and Git. Download for personal research under the provider's applicable terms. The packaged outputs are research-derived forecasts, aggregate results, features and source provenance; no raw quote history is redistributed. Do not imply unrestricted data licensing.

## Macroeconomic vintages

Sources: UNRATE, CPIAUCSL, FEDFUNDS. The public ALFRED graph CSV download accepts a requested vintage_date for each series. This is a website download endpoint, not a documented stable API contract. Every requested series/date header is checked; a missing header or future nonblank observation raises an error. No current-vintage or fixed-lag substitution is permitted.

Requests use batches of four vintage dates (12 series columns), pause between new requests and reuse cached raw responses. The feature table has one row per month-end from December 2005 through December 2025: 241 snapshots. Each snapshot is usable from the following calendar day. CPI inflation uses two observations inside the same vintage; unemployment changes use the last four available observations inside it. This handles revisions using what the selected historical vintage actually contained.

Official ALFRED help explains vintage dates and observation/revision availability. The endpoint itself is validated empirically by the downloader. A changed endpoint must be repaired explicitly rather than silently creating look-ahead bias.

## Reproducibility and limits

The config hash and full validation search are recorded before results are assessed. Source hashes document what was used. Packages and the dataset calendar are documented. The three-ETF universe is chosen retrospectively and does not eliminate survivorship/selection effects. Reference output files are not proof of a live, prospective or independent out-of-sample trading process.

## Source references

- https://alfred.stlouisfed.org/help/downloaddata
- https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- https://finance.yahoo.com/
