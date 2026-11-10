import numpy as np
import pandas as pd
from macroquant.metrics import portfolio_statistics

def risky_weights(variances,correlation,target_vol,max_weight):
    volatility=np.sqrt(np.maximum(np.asarray(variances,dtype=float),1e-8))
    base=1/volatility;base=base/base.sum();base=np.minimum(base,max_weight)
    covariance=np.outer(volatility,volatility)*np.asarray(correlation)
    expected=float(np.sqrt(max(base@covariance@base,1e-12)))
    scale=min(1,target_vol/expected)
    result=base*scale
    if (result<0).any() or result.sum()>1+1e-10 or (result>max_weight+1e-10).any():raise AssertionError("Portfolio constraint violation")
    return result

def backtest(prices,panel,predictions,config,cost_bps,strategy="forecast"):
    assets=config["assets"];calendar=prices[assets[0]].index;h=config["horizon_sessions"]
    opens=pd.DataFrame({asset:prices[asset].adjusted_open for asset in assets})
    closes=pd.DataFrame({asset:prices[asset].adjusted_close for asset in assets})
    dates=sorted(predictions.date.unique());previous=np.zeros(len(assets));daily=[];trades=[]
    rate=cost_bps/10000
    for counter,date in enumerate(dates):
        date=pd.Timestamp(date);rows=panel.loc[panel.date==date].set_index("asset").loc[assets]
        position=int(rows.position.iloc[0]);entry=position+1;exit_position=position+h+1
        if counter and pd.Timestamp(trades[-1]["exit_date"])!=calendar[entry]:raise AssertionError("Holding intervals must join without gaps or overlaps")
        if strategy=="buy_hold":weights=np.full(len(assets),1/len(assets)) if counter==0 else previous.copy()
        elif strategy=="equal_weight":weights=np.full(len(assets),1/len(assets))
        else:
            values=predictions.loc[predictions.date==date].set_index("asset").loc[assets].forecast_variance.to_numpy()
            known=np.log(closes.iloc[max(0,position-251):position+1]).diff().dropna()
            correlation=known.corr().fillna(0).to_numpy();np.fill_diagonal(correlation,1)
            weights=risky_weights(values,correlation,config["target_portfolio_volatility"],config["max_asset_weight"])
        turnover=float(np.sum(np.abs(weights-previous)));cost=turnover*rate
        carried=weights.copy()
        for j in range(entry+1,exit_position+1):
            asset_returns=(opens.iloc[j]/opens.iloc[j-1]-1).to_numpy()
            gross=float(carried@asset_returns)
            daily_cost=cost if j==entry+1 else 0
            net=(1-daily_cost)*(1+gross)-1
            daily.append({"date":calendar[j],"net_return":net,"gross_return":gross,"cost_fraction":daily_cost,"decision_date":date})
            carried=carried*(1+asset_returns)/(1+gross)
        previous=carried
        trades.append({"decision_date":date,"entry_date":calendar[entry],"exit_date":calendar[exit_position],"turnover_l1":turnover,"initial_cost_fraction":cost,**{asset+"_weight":float(w) for asset,w in zip(assets,weights)},"cash_weight":float(1-weights.sum())})
    # Include the cost of closing remaining risky positions at the final open.
    final_cost=rate*float(previous.sum())
    daily[-1]["net_return"]=(1+daily[-1]["net_return"])*(1-final_cost)-1
    daily[-1]["cost_fraction"]+=final_cost
    daily=pd.DataFrame(daily);daily["nav"]=(1+daily.net_return).cumprod()
    daily.attrs["inception_date"]=trades[0]["entry_date"]
    statistics=portfolio_statistics(daily)
    statistics.update(cost_bps=cost_bps,mean_period_turnover=float(np.mean([r["turnover_l1"] for r in trades])),liquidation_cost_fraction=final_cost)
    return daily,pd.DataFrame(trades),statistics
