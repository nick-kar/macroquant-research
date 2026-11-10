import html,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CSS="body{margin:0;background:#eef1f5;color:#1b2b40;font:15px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:36px auto;background:white;padding:46px 52px;border-top:6px solid #1d566f;box-shadow:0 4px 30px #13263910}h1{font-size:36px;line-height:1.15;margin:10px 0 18px}h2{margin-top:38px;border-bottom:1px solid #dce3eb;padding-bottom:9px;font-size:22px}h3{font-size:17px;color:#265a70}.eyebrow{letter-spacing:2px;font-size:11px;color:#547384}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{background:#f0f5f8;padding:14px 20px;min-width:190px}.card strong{display:block;font-size:23px;color:#1d566f}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:right;padding:9px;border-bottom:1px solid #e1e6ed}th:first-child,td:first-child{text-align:left}th{background:#edf3f7}img{width:100%;height:auto;margin:15px 0}code{background:#edf1f5;padding:2px 5px}.note{border-left:4px solid #b08840;padding:12px 18px;background:#fcf8ef}footer{color:#64748b;font-size:12px;margin-top:40px}@media(max-width:700px){main{padding:25px 18px;margin:0}h1{font-size:28px}}"

def table(frame):return '<div class="table">'+frame.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}",escape=True)+'</div>'

def report(config,output):
    output=Path(output);figures=output/"figures";figures.mkdir(exist_ok=True)
    summary=json.loads((output/"summary.json").read_text());selection=json.loads((output/"selection.json").read_text())
    validation=pd.read_csv(output/"validation_metrics.csv");test=pd.read_csv(output/"test_metrics.csv");contrasts=pd.read_csv(output/"macro_ablation_tests.csv");portfolio=pd.read_csv(output/"portfolio_metrics.csv");paths=pd.read_csv(output/"portfolio_daily_paths.csv",parse_dates=["date"])
    plt.rcParams.update({"figure.dpi":130,"axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":"#233c50","text.color":"#233c50","font.size":10,"axes.grid":True,"grid.alpha":.2})
    names=test.model.tolist();x=np.arange(len(names));fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(x-.18,validation.set_index("model").loc[names].qlike,.36,label="Validation",color="#aabfca")
    ax.bar(x+.18,test.set_index("model").loc[names].qlike,.36,label="Chronological test",color="#1d566f")
    ax.set_xticks(x,names,rotation=20,ha="right");ax.set_ylabel("Mean QLIKE (lower is better)");ax.legend();fig.tight_layout();fig.savefig(figures/"forecast_losses.png");plt.close(fig)
    champion=summary["validation_champion"];shown=list(dict.fromkeys([champion,"rv20","equal_weight","buy_hold"]))
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name in shown:
        rows=paths.loc[paths.model==name];ax.plot(rows.date,rows.nav,label=name,linewidth=1.7)
    ax.set_title(f"Historical adjusted-price proxy portfolios · {config['main_cost_bps']} bps one-way cost")
    ax.set_ylabel("Net wealth index, initial 1");ax.legend();fig.tight_layout();fig.savefig(figures/"portfolio_paths.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.4));positions=np.arange(len(contrasts))
    ax.errorbar(contrasts.mean_difference,positions,xerr=[contrasts.mean_difference-contrasts.ci_low,contrasts.ci_high-contrasts.mean_difference],fmt="o",color="#1d566f",capsize=5)
    ax.axvline(0,color="#a67835",linestyle="--");ax.set_yticks(positions,contrasts.comparison);ax.set_xlabel("Macro minus price-only QLIKE · 95% moving-block bootstrap interval");fig.tight_layout();fig.savefig(figures/"macro_ablation.png");plt.close(fig)
    main=portfolio.loc[portfolio.cost_bps==config["main_cost_bps"]].copy()
    for col in ["cagr","annualized_volatility","max_drawdown"]:main[col]=main[col]*100
    main=main.rename(columns={"cagr":"CAGR %","annualized_volatility":"Volatility %","max_drawdown":"Max drawdown %"})
    sections=[]
    for _,row in contrasts.iterrows():
        direction="lower" if row.mean_difference<0 else "higher"
        evidence="excludes zero" if row.ci_high<0 or row.ci_low>0 else "includes zero"
        sections.append(f"{html.escape(row.comparison)}: the estimated mean loss difference is {row.mean_difference:.4f} ({direction} loss with macro features); its 95% block-bootstrap interval [{row.ci_low:.4f}, {row.ci_high:.4f}] {evidence}. Holm-adjusted p={row.holm_adjusted_p:.4f} across the two predeclared macro contrasts.")
    interpretation=''.join('<p>'+s+'</p>' for s in sections)
    content=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MacroQuant research report</title><style>{CSS}</style></head><body><main>
<div class="eyebrow">MACROQUANT · REPRODUCIBLE EMPIRICAL RESEARCH</div><h1>Do macroeconomic vintages improve<br>market-risk forecasts?</h1>
<p>A chronological comparison of statistical and machine-learning forecasts, with next-session execution and explicit trading-cost assumptions.</p>
<div class="cards"><div class="card"><strong>{summary['market_sessions']:,}</strong>common market sessions</div><div class="card"><strong>{summary['test_periods']}</strong>non-overlapping test periods</div><div class="card"><strong>{champion}</strong>selected on validation</div></div>
<h2>1. Research design</h2><p>Universe: SPY, TLT, GLD. Market history {summary['market_first']} to {summary['market_last']}. Signals every five trading sessions after a 252-session warm-up. Macroeconomic inputs use dated ALFRED snapshots, conservatively available from the day after each vintage date. No current-vintage substitution is allowed.</p>
<p>Features are observed at close t. Positions are assumed entered at open t+1 and held until open t+6. The forecast target is annualized mean squared log adjusted-open returns over those five held intervals. Learned models train only on labels ending strictly before their refit date. Hyperparameters and the champion are chosen on 2014–2018 validation; prequential test decisions begin in 2019. Refit every 13 signal periods.</p>
<p class="note">This is retrospective research, not a prospectively registered strategy or live trading track record. Adjusted open is a research return proxy derived from Yahoo's adjustment factor, not an executable historical share price. A three-ETF universe chosen today does not remove selection bias. No raw market prices are redistributed with this report.</p>
<h2>2. Forecast evaluation</h2><h3>Validation</h3>{table(validation)}<h3>Chronological test</h3>{table(test)}<img src="figures/forecast_losses.png" alt="Validation and test forecast losses">
<h2>3. Does macro information help?</h2>{interpretation}{table(contrasts)}<img src="figures/macro_ablation.png" alt="Macro feature ablation uncertainty intervals"><p>Bootstrap units are signal dates, averaging loss across assets before resampling to preserve cross-asset dependence. Circular moving blocks contain 13 five-session periods. The two focal macro comparisons receive Holm adjustment; this does not correct for every possible strategy previously researched.</p>
<h2>4. Portfolio implementation</h2><p>Forecast-based allocations use inverse predicted volatility, a 60% asset cap, a 10% forecast portfolio-volatility target, no leverage, and zero interest on residual cash. Covariance correlations use only the preceding 252 closes. Costs are applied to the L1 risky-weight change, including initial entry and final liquidation. Holdings drift between scheduled rebalances. Equal-weight rebalancing and buy-and-hold are comparators, not equal-risk guarantees.</p>{table(main[['model','CAGR %','Volatility %','sharpe_zero_cash','Max drawdown %','mean_period_turnover']])}<img src="figures/portfolio_paths.png" alt="Historical net portfolio wealth paths">
<h3>Cost sensitivity for the validation-selected champion</h3>{table(portfolio.loc[portfolio.model==champion,['model','cost_bps','cagr','annualized_volatility','sharpe_zero_cash','max_drawdown']])}
<h2>5. Interpretation and limits</h2><p>The research conclusion follows the measured loss differences above. A favourable backtest is not proof of an exploitable signal or future performance; a failure to outperform is a valid research outcome. Allocation risk differs across models and benchmarks. Daily open-based drawdowns omit within-session extremes, cash earns zero, and the transaction-cost model omits size-dependent market impact, taxes and broker-specific fills.</p><p>Small five-day realized-variance labels are noisy. ETF selection, free-data corrections and limited hyperparameter search remain material limitations. Macroeconomic vintage availability is checked at daily rather than intraday granularity. GARCH fallback refits: {summary['garch_fallback_refits']}. Paper tracking is separate and has no historical live-performance claim.</p>
<h2>6. Reproduce and audit</h2><p>Inspect experiment_spec.json, selection.json, provenance.json, validation_search_log.csv, training_audit.csv and the per-date prediction/weight files. Run <code>python -m macroquant run</code> from your own downloaded source cache. Source changes can change results; raw response hashes are recorded. The configuration was written before this run and was not selected using test performance.</p><footer>Generated {html.escape(summary['completed_utc'])}. Research software and historical simulation; no broker connection or real-money trades.</footer></main></body></html>"""
    (output/"report.html").write_text(content,encoding="utf-8")
    md=f"# MacroQuant research findings\n\nValidation-selected champion: **{champion}**.\n\nTest: {summary['test_periods']} five-session periods across three ETFs.\n\n"+"\n\n".join(s.replace("&gt;",">") for s in sections)+"\n\nSee report.html and source CSVs for all models, costs and limitations. No live trading performance is claimed.\n"
    (output/"findings.md").write_text(md)
