import numpy as np
import pandas as pd

def qlike(actual,forecast):
    ratio=np.maximum(np.asarray(actual),1e-10)/np.maximum(np.asarray(forecast),1e-8)
    return ratio-np.log(ratio)-1

def forecast_metrics(joined):
    rows=[]
    for model,group in joined.groupby("model"):
        actual=group.target_variance.to_numpy();pred=group.forecast_variance.to_numpy()
        rows.append({"model":model,"observations":len(group),"qlike":float(np.mean(qlike(actual,pred))),"variance_mse":float(np.mean((actual-pred)**2)),"volatility_mae":float(np.mean(np.abs(np.sqrt(actual)-np.sqrt(pred))))})
    return pd.DataFrame(rows).sort_values("qlike")

def block_bootstrap_mean(values,repetitions=1000,block=13,seed=42):
    values=np.asarray(values,dtype=float)
    if len(values)<2:raise ValueError("Need at least two observations")
    rng=np.random.default_rng(seed);n=len(values);blocks=int(np.ceil(n/block));samples=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=blocks)
        indexes=np.concatenate([(np.arange(s,s+block)%n) for s in starts])[:n]
        samples.append(values[indexes].mean())
    samples=np.array(samples);mean=float(values.mean())
    null=samples-mean
    return {"mean_difference":mean,"ci_low":float(np.quantile(samples,.025)),"ci_high":float(np.quantile(samples,.975)),"p_two_sided":float((1+np.sum(np.abs(null)>=abs(mean)))/(repetitions+1)),"periods":n,"block_periods":block,"bootstrap_repetitions":repetitions}

def macro_tests(joined,config):
    records=[]
    for family in ["ridge","hgb"]:
        losses={}
        for suffix in ["price","macro"]:
            group=joined.loc[joined.model==family+"_"+suffix].copy()
            group["loss"]=qlike(group.target_variance,group.forecast_variance)
            losses[suffix]=group.groupby("date").loss.mean()
        aligned=pd.concat(losses,axis=1).dropna()
        result=block_bootstrap_mean(aligned["macro"]-aligned["price"],config["bootstrap_repetitions"],config["bootstrap_block_periods"],config["random_seed"])
        records.append({"comparison":family+": macro minus price QLIKE",**result})
    # Two predeclared focal comparisons; adjusted p-values control this family only.
    order=np.argsort([r["p_two_sided"] for r in records]);running=0
    for rank,index in enumerate(order):
        running=max(running,min(1,records[index]["p_two_sided"]*(len(records)-rank)))
        records[index]["holm_adjusted_p"]=running
    return pd.DataFrame(records)

def portfolio_statistics(daily):
    r=daily.net_return.to_numpy();nav=np.cumprod(1+r);dates=pd.to_datetime(daily.date)
    inception=pd.Timestamp(daily.attrs.get("inception_date",dates.iloc[0]))
    years=max((dates.iloc[-1]-inception).days/365.25,1/252)
    volatility=float(np.std(r,ddof=1)*np.sqrt(252))
    return {"cagr":float(nav[-1]**(1/years)-1),"annualized_volatility":volatility,"sharpe_zero_cash":float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(252)) if volatility else 0,"max_drawdown":float(np.min(nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1)),"terminal_nav":float(nav[-1]),"observations":len(r)}
