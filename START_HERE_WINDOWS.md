# MacroQuant — start here on Windows

This is a new, separate project. Keep your previous SQL, predictive maintenance and UpgradeLens files unchanged. No Docker, broker account, LLM key or paid data subscription is required for the included workflow. Historical downloads use public endpoints and can be temporarily unavailable.

## 1. Inspect the completed reference research
Extract the ZIP. Open the **macroquant** folder containing this file and requirements.txt. Before installing anything, open **reference_results/report.html** in your browser. It contains the executed reference results, plots, comparisons and limitations. Keep the figures subfolder alongside report.html.

These are genuine historical-data research results produced during packaging, not promised profits or simulated synthetic prices. You must rerun and understand the project before presenting its execution as your own work.

## 2. Open in VS Code and install
Open the same folder in VS Code. Choose Terminal > New Terminal. Run separately:
```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Use your already installed Python 3.12.10. No environment activation or PowerShell policy change is required. The tests use small synthetic fixtures for software checks; research results use downloaded real data.

## 3. Reproduce the research
```powershell
.\.venv\Scripts\python.exe -m macroquant run --output results/my_first_run
```
First run downloads market history and 241 month-end macro vintages, cached under data/cache. Allow several minutes for the network downloads and model search. Progress prints market counts, vintage counts and validation-selected parameters. Do not interrupt simply because a fit is quiet for a short period.

The public ALFRED graph endpoint returns limited numbers of series per request. The code requests small batches, validates every vintage header, and refuses to replace a missing vintage with today's data. It pauses between new batches. Requests are cached, so a resumed download does not start from zero.

Expected final output: a JSON summary plus **results/my_first_run/report.html**. Open that HTML file to review your own result. A complete run cannot be overwritten; use a new output folder for any intentional rerun. Do not repeatedly change settings to obtain a more attractive test result.

Raw market prices are not included in the ZIP; download them for your own research and follow the provider's terms. Do not upload data/cache or other raw prices to GitHub. Reference results include derived research outputs and source hashes, not raw price histories.

## 4. The short CV acceptance checklist
- Tests pass on your PC.
- Your full run completes.
- You can explain the feature/label/entry timing in docs/RESEARCH_PROTOCOL.md.
- You can state the actual macro-ablation conclusion without changing it to sound more successful.
- Your repository has a clear README, source code and reference report.

Then use the verified CV wording in docs/CV_AND_INTERVIEW.md. A paper-tracking step is optional and separate.

## 5. Optional forward forecast logging
After reviewing the reference selection:
```powershell
.\.venv\Scripts\python.exe -m macroquant paper --selection reference_results/selection.json
```
This downloads data through yesterday, computes a current forecast with the validation-selected method, and appends a dated record to paper/forecasts.sqlite. It does not send orders, connect to a broker, or create a live performance history. The same signal date cannot be overwritten by this command.

## Troubleshooting
- No module named macroquant: terminal must be in the folder containing requirements.txt and the macroquant package directory.
- Yahoo or ALFRED unavailable: wait and retry. No fake data are substituted. Preserve valid cached downloads.
- Missing vintage header: the endpoint contract changed or truncated the requested columns. Stop; do not relabel the data as point-in-time.
- Completed run exists: choose a new output directory.
- Small numerical differences: compare package versions and raw source hashes first. Historical providers can revise data.
- No paper record written: source may be stale or the signal date was already logged. This is deliberate rather than silently editing a forecast.
