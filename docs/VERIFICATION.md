# Executed verification — 10 November 2026

Executed in a clean Linux Python 3.12.14 environment with the pinned requirements. Windows instructions target Python 3.12; Windows execution is still for the recipient to verify.

- 29 automated tests passed. Exact output: reference_results/test_results.txt.
- Seven groups of internal timing/accounting checks passed: reference_results/audit_results.json.
- Real historical market data: SPY, TLT, GLD; 5,283 common sessions, 3 January 2005–31 December 2025.
- Point-in-time macro inputs: 241 month-end vintages, December 2005–December 2025. Source URLs and SHA-256 hashes are recorded in provenance.json.
- Seven forecasting methods; 251 validation decision dates and 350 test decision dates (1,050 asset-level test forecasts per method).
- GARCH selected on validation. EWMA had the lowest test QLIKE (0.468719), compared with GARCH (0.478929). The held-out winner was not substituted for the validation selection.
- Adding macro features increased estimated mean QLIKE loss by 0.026235 for Ridge and 0.021709 for histogram gradient boosting. Both 95% block-bootstrap intervals included zero; Holm-adjusted p-values were 0.5994. This experiment does not establish incremental macro predictive value.
- Costs evaluated at 0, 5, 10 and 25 basis points one-way, including entry and liquidation. Daily portfolio paths and weights are included.
- GARCH nonconvergence fallbacks in this reference run: zero.

The reference report contains measured historical results, not a live track record. Optional forward forecast logging has unit-test coverage but was not executed against live endpoints during this packaging run. No broker execution, cloud deployment, independent price-feed reconciliation or commercial operation is claimed. The GitHub workflow is provided but has not yet run in the recipient's repository.

The initial implementation was corrected to purge validation labels crossing the first test signal before producing this final reference run. Tests explicitly cover that boundary. Data revisions, adjusted-price approximations, a retrospectively chosen ETF universe and the small universe limit the economic conclusions. See RESEARCH_PROTOCOL.md and DATA_AND_PROVENANCE.md.
