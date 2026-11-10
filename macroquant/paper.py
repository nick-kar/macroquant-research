import hashlib,json,sqlite3
from contextlib import closing
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from macroquant.data import load_data
from macroquant.features import make_panel,price_features,macro_asof,eligible_training
from macroquant.models import model_pipeline,garch_forecasts
from macroquant.backtest import risky_weights

def paper(config,selection_path,cache,ledger):
    selection=json.loads(Path(selection_path).read_text())
    today=datetime.now(timezone.utc).date();end=str(today-timedelta(days=1))
    prices,macro,provenance=load_data(config,cache,end=end)
    date=next(iter(prices.values())).index[-1]
    if (pd.Timestamp(today)-date).days>7:raise ValueError("Latest source price is stale; not logging a forward forecast")
    history=make_panel(prices,macro,config,include_unlabelled=True)
    values=macro_asof(macro,[date]).iloc[0].to_dict();rows=[]
    for asset,frame in prices.items():rows.append({"date":date,"asset":asset,**price_features(frame).loc[date].to_dict(),**values})
    current=pd.DataFrame(rows);name=selection["validation_champion"]
    if current.isna().any().any():raise ValueError("Missing current features")
    with threadpool_limits(limits=1):
        if name in {"rv20","ewma"}:forecast=current[name].to_numpy()
        elif name=="garch":
            combined=pd.concat([history,current],ignore_index=True).drop_duplicates(["date","asset"],keep="last")
            pred,_=garch_forecasts(prices,combined,[date],config)
            forecast=pred.set_index("asset").loc[config["assets"]].forecast_variance.to_numpy()
        else:
            training=eligible_training(history,date);model=model_pipeline(name,selection["parameters"][name],config["random_seed"])
            model.fit(training,training.target_log_variance);forecast=np.exp(np.clip(model.predict(current),-18,4))
    close=pd.DataFrame({asset:prices[asset].adjusted_close for asset in config["assets"]})
    correlation=np.log(close.tail(252)).diff().dropna().corr().to_numpy()
    weights=risky_weights(forecast,correlation,config["target_portfolio_volatility"],config["max_asset_weight"])
    payload={"created_utc":datetime.now(timezone.utc).isoformat(),"signal_date":str(date.date()),"model":name,"forecast_volatility":dict(zip(config["assets"],np.sqrt(forecast).tolist())),"target_weights":dict(zip(config["assets"],weights.tolist())),"cash_weight":float(1-weights.sum()),"status":"forward research forecast; no orders submitted","source_hashes":[record["sha256"] for record in provenance["market"]]}
    append_forecast(payload,ledger)
    return payload


def append_forecast(payload,ledger):
    Path(ledger).parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(ledger)) as db,db:
        db.execute("CREATE TABLE IF NOT EXISTS forecasts (signal_date TEXT PRIMARY KEY,payload TEXT,previous_hash TEXT,record_hash TEXT)")
        last=db.execute("SELECT signal_date,record_hash FROM forecasts ORDER BY signal_date DESC LIMIT 1").fetchone()
        if last and payload["signal_date"]<=last[0]:raise ValueError("A forecast exists for this date or later; existing records cannot be overwritten or backdated")
        previous=last[1] if last else "GENESIS"
        raw=json.dumps(payload,sort_keys=True);hash_value=hashlib.sha256((previous+raw).encode()).hexdigest()
        db.execute("INSERT INTO forecasts VALUES (?,?,?,?)",(payload["signal_date"],raw,previous,hash_value))
    return hash_value
