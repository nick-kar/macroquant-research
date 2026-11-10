import hashlib,json,platform,time,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from macroquant.data import load_data
from macroquant.features import make_panel,split_dates
from macroquant.models import MODEL_NAMES,walk_forward_ml,garch_forecasts
from macroquant.metrics import forecast_metrics,qlike,macro_tests
from macroquant.backtest import backtest

ROOT=Path(__file__).resolve().parents[1]

def config_load(path=None):return json.loads(Path(path or ROOT/"config/research.json").read_text())

def join_forecasts(forecasts,panel):
    return forecasts.merge(panel[["date","asset","target_variance","label_end","entry_date","exit_date","asset_return"]],on=["date","asset"],validate="many_to_one")

def run(config,cache,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec={"config":config,"config_sha256":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"started_utc":datetime.now(timezone.utc).isoformat(),"protocol":"Retrospective predefined chronological validation and prequential test. No hyperparameters selected using test losses. Not a prospectively registered strategy."}
    (output/"experiment_spec.json").write_text(json.dumps(spec,indent=2))
    prices,macro,provenance=load_data(config,cache)
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2))
    macro.to_csv(output/"macro_features_by_vintage.csv",index=False)
    panel=make_panel(prices,macro,config)
    if panel.empty:raise ValueError("Empty research panel")
    panel.to_csv(output/"research_panel.csv",index=False)
    validation_dates,test_dates=split_dates(panel,config)
    candidates=[];selected={};audits=[];validation_parts=[]
    with threadpool_limits(limits=1):
        for name in ["ridge_price","ridge_macro","hgb_price","hgb_macro"]:
            parameters=config["ridge_alphas"] if name.startswith("ridge") else config["hgb_leaf_nodes"]
            results=[]
            for parameter in parameters:
                forecast,audit=walk_forward_ml(panel,validation_dates,name,parameter,config)
                metric=forecast_metrics(join_forecasts(forecast,panel)).iloc[0].to_dict()
                results.append((metric["qlike"],parameter,forecast,audit));candidates.append({**metric,"parameter":parameter})
            loss,parameter,forecast,audit=min(results,key=lambda row:row[0]);selected[name]=parameter
            validation_parts.append(forecast);audits.extend(audit)
            print("Selected on validation:",name,parameter,"QLIKE",round(loss,4),flush=True)
        for name in ["rv20","ewma"]:
            part=panel.loc[panel.date.isin(validation_dates),["date","asset",name]].rename(columns={name:"forecast_variance"});part["model"]=name;validation_parts.append(part)
        garch,audit=garch_forecasts(prices,panel,validation_dates,config);validation_parts.append(garch);audits.extend(audit)
        validation=join_forecasts(pd.concat(validation_parts,ignore_index=True),panel)
        validation_metrics=forecast_metrics(validation);champion=validation_metrics.iloc[0]["model"]
        selection={"parameters":selected,"validation_champion":champion,"selection_metric":"mean validation QLIKE","test_used_for_selection":False}
        (output/"selection.json").write_text(json.dumps(selection,indent=2))
        test_parts=[]
        for name,parameter in selected.items():
            forecast,audit=walk_forward_ml(panel,test_dates,name,parameter,config);test_parts.append(forecast);audits.extend(audit)
        for name in ["rv20","ewma"]:
            part=panel.loc[panel.date.isin(test_dates),["date","asset",name]].rename(columns={name:"forecast_variance"});part["model"]=name;test_parts.append(part)
        garch,audit=garch_forecasts(prices,panel,test_dates,config);test_parts.append(garch);audits.extend(audit)
    validation.to_csv(output/"validation_predictions.csv",index=False)
    validation_metrics.to_csv(output/"validation_metrics.csv",index=False)
    pd.DataFrame(candidates).to_csv(output/"validation_search_log.csv",index=False)
    test=join_forecasts(pd.concat(test_parts,ignore_index=True),panel)
    test.to_csv(output/"test_predictions.csv",index=False)
    test_metrics=forecast_metrics(test);test_metrics.to_csv(output/"test_metrics.csv",index=False)
    contrasts=macro_tests(test,config);contrasts.to_csv(output/"macro_ablation_tests.csv",index=False)
    test["qlike"]=qlike(test.target_variance,test.forecast_variance)
    test["year"]=pd.to_datetime(test.date).dt.year
    test.groupby(["model","year"]).qlike.mean().unstack("year").to_csv(output/"loss_by_year.csv")
    pd.DataFrame(audits).to_csv(output/"training_audit.csv",index=False)
    statistics=[];paths=[]
    for name in MODEL_NAMES+["equal_weight","buy_hold"]:
        for cost in config["one_way_cost_bps"]:
            predictions=test.loc[test.model==(name if name in MODEL_NAMES else "rv20")]
            daily,trades,metrics=backtest(prices,panel,predictions,config,cost,name if name in {"equal_weight","buy_hold"} else "forecast")
            statistics.append({"model":name,**metrics})
            if cost==config["main_cost_bps"]:
                daily["model"]=name;paths.append(daily);trades.to_csv(output/("weights_"+name+".csv"),index=False)
    pd.DataFrame(statistics).to_csv(output/"portfolio_metrics.csv",index=False)
    pd.concat(paths,ignore_index=True).to_csv(output/"portfolio_daily_paths.csv",index=False)
    import importlib.metadata as md
    versions={name:md.version(name) for name in ["numpy","pandas","scikit-learn","arch","scipy"]}
    summary={"assets":config["assets"],"market_first":str(next(iter(prices.values())).index.min().date()),"market_last":str(next(iter(prices.values())).index.max().date()),"market_sessions":len(next(iter(prices.values()))),"panel_rows":len(panel),"validation_periods":len(validation_dates),"test_periods":len(test_dates),"test_first_signal":str(test_dates[0].date()),"test_last_signal":str(test_dates[-1].date()),"validation_champion":champion,"python":platform.python_version(),"versions":versions,"completed_utc":datetime.now(timezone.utc).isoformat(),"garch_fallback_refits":sum(bool(a.get("fallback",False)) for a in audits)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    from macroquant.report import report
    report(config,output)
    return summary
