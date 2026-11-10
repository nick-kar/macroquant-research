import hashlib,io,json,time,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

class DataError(RuntimeError):pass

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def get(url,path,attempts=3,pause=1):
    path=Path(path)
    if path.exists():return path.read_bytes()
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(attempts):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"MacroQuant academic personal research/1.0"})
            with urllib.request.urlopen(request,timeout=45) as response:raw=response.read(15_000_001)
            if len(raw)>15_000_000:raise DataError("Source response exceeded 15 MB")
            if not raw:raise DataError("Empty source response")
            temporary=path.with_suffix(path.suffix+".tmp");temporary.write_bytes(raw);temporary.replace(path)
            return raw
        except Exception:
            if attempt==attempts-1:raise
            time.sleep(pause*(attempt+1))

def market_download(asset,start,end,cache):
    start_seconds=int(pd.Timestamp(start,tz="UTC").timestamp())
    end_seconds=int((pd.Timestamp(end,tz="UTC")+pd.Timedelta(days=1)).timestamp())
    params=urllib.parse.urlencode({"period1":start_seconds,"period2":end_seconds,"interval":"1d","events":"div,splits"})
    url="https://query1.finance.yahoo.com/v8/finance/chart/"+asset+"?"+params
    path=Path(cache)/"prices"/f"{asset}_{start}_{end}.json"
    raw=get(url,path)
    try:obj=json.loads(raw)["chart"]["result"][0]
    except (ValueError,TypeError,KeyError,IndexError) as exc:raise DataError("Invalid historical market response; no synthetic fallback") from exc
    times=obj.get("timestamp",[]);quote=obj["indicators"]["quote"][0]
    if "adjclose" not in obj["indicators"]:raise DataError("Adjusted close missing")
    adj=obj["indicators"]["adjclose"][0]["adjclose"]
    frame=pd.DataFrame({"date":pd.to_datetime(times,unit="s",utc=True).tz_convert(None).normalize(),"open_raw":quote["open"],"close_raw":quote["close"],"adjusted_close":adj,"volume":quote.get("volume")})
    frame=frame.dropna(subset=["open_raw","close_raw","adjusted_close"]).set_index("date").sort_index()
    frame=frame.loc[start:end]
    if frame.index.has_duplicates:raise DataError("Duplicate market session")
    if len(frame)<500:raise DataError("Insufficient market history")
    if (frame[["open_raw","close_raw","adjusted_close"]]<=0).any().any():raise DataError("Nonpositive market price")
    frame["adjustment_factor"]=frame.adjusted_close/frame.close_raw
    frame["adjusted_open"]=frame.open_raw*frame.adjustment_factor
    return frame,{"asset":asset,"url":url,"cache_file":str(path),"sha256":sha(path),"retrieved_utc":datetime.now(timezone.utc).isoformat(),"rows":len(frame),"first":str(frame.index.min().date()),"last":str(frame.index.max().date()),"basis":"Yahoo adjusted-close factor applied to raw open; research total-return proxy, not an executable adjusted share price"}

def month_ends(start,end):return [d.strftime("%Y-%m-%d") for d in pd.date_range(start,end,freq="ME")]

def parse_alfred(raw,series,vintage):
    data=pd.read_csv(io.BytesIO(raw),na_values=[".",""])
    expected=series+"_"+vintage.replace("-","")
    if expected not in data.columns:raise DataError("ALFRED did not confirm requested vintage: "+expected)
    date_col=data.columns[0]
    result=pd.Series(pd.to_numeric(data[expected],errors="coerce").values,index=pd.to_datetime(data[date_col]),name=series).dropna().sort_index()
    # A malformed endpoint must not make future observations available.
    if (result.index>pd.Timestamp(vintage)).any():raise DataError("Vintage contains observations after its as-of date")
    return result

def macro_download(config,cache,end=None):
    end=end or config["data_end"];dates=month_ends(config["macro_start"],end)
    rows=[];provenance=[];series=config["macro_series"]
    for offset in range(0,len(dates),4):
        batch=dates[offset:offset+4];ids=[];vintages=[]
        for date in batch:
            for ident in series:ids.append(ident);vintages.append(date)
        params=urllib.parse.urlencode({"id":",".join(ids),"vintage_date":",".join(vintages)})
        # The graph CSV is a public download endpoint, not a guaranteed API contract.
        # Each requested vintage header is validated. No current-vintage substitution.
        key=hashlib.sha256(params.encode()).hexdigest()[:16]
        path=Path(cache)/"macro"/("vintages_"+key+".csv")
        existed=path.exists()
        raw=get("https://alfred.stlouisfed.org/graph/alfredgraph.csv?"+params,path)
        for date in batch:
            values={ident:parse_alfred(raw,ident,date) for ident in series}
            unrate=values["UNRATE"];cpi=values["CPIAUCSL"];rate=values["FEDFUNDS"]
            if any(v.empty for v in values.values()):raise DataError("Empty vintage history")
            latest_cpi=cpi.index[-1];year_ago=latest_cpi-pd.DateOffset(years=1)
            if year_ago not in cpi.index or len(unrate)<4:raise DataError("Insufficient vintage history for features")
            row={"vintage_date":date,"available_date":str((pd.Timestamp(date)+pd.Timedelta(days=1)).date()),
                 "unemployment":float(unrate.iloc[-1]),"unemployment_change_3m":float(unrate.iloc[-1]-unrate.iloc[-4]),
                 "inflation_yoy":float(100*(cpi.iloc[-1]/cpi.loc[year_ago]-1)),"policy_rate":float(rate.iloc[-1]),
                 "unemployment_observation":str(unrate.index[-1].date()),"inflation_observation":str(cpi.index[-1].date()),"rate_observation":str(rate.index[-1].date())}
            rows.append(row)
        provenance.append({"source":"ALFRED public graph CSV","requested_vintages":batch,"series":series,"sha256":sha(path),"cache_file":str(path),"url":"https://alfred.stlouisfed.org/graph/alfredgraph.csv?"+params})
        if not existed:time.sleep(1)
        print(f"ALFRED vintages: {min(offset+4,len(dates))}/{len(dates)}",flush=True)
    frame=pd.DataFrame(rows).sort_values("available_date")
    return frame,provenance

def load_data(config,cache,end=None):
    end=end or config["data_end"];prices={};records=[]
    for asset in config["assets"]:
        frame,record=market_download(asset,config["data_start"],end,cache);prices[asset]=frame;records.append(record)
        print(asset,len(frame),"market sessions",flush=True)
    common=prices[config["assets"][0]].index
    for frame in prices.values():common=common.intersection(frame.index)
    dropped={asset:len(frame)-len(common) for asset,frame in prices.items()}
    prices={asset:frame.loc[common].copy() for asset,frame in prices.items()}
    macro,provenance=macro_download(config,cache,end)
    return prices,macro,{"market":records,"macro":provenance,"calendar_intersection_dropped_rows":dropped,"fetched_utc":datetime.now(timezone.utc).isoformat()}
