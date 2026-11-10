import numpy as np
import pandas as pd
PRICE_FEATURES=["log_rv5","log_rv20","log_rv60","momentum20","momentum60","downside20"]
MACRO_FEATURES=["unemployment","unemployment_change_3m","inflation_yoy","policy_rate"]

# All feature calculations are backward-looking and scale invariant.
def price_features(frame):
    close_return=np.log(frame.adjusted_close).diff()
    open_return=np.log(frame.adjusted_open).diff()
    features=pd.DataFrame(index=frame.index)
    for horizon in [5,20,60]:features[f"log_rv{horizon}"]=np.log((open_return.pow(2).rolling(horizon).mean()*252).clip(lower=1e-8))
    features["momentum20"]=close_return.rolling(20).sum()
    features["momentum60"]=close_return.rolling(60).sum()
    features["downside20"]=close_return.clip(upper=0).pow(2).rolling(20).mean()*252
    features["rv20"]=np.exp(features.log_rv20)
    # Recursive EWMA includes only returns known at the decision date.
    features["ewma"]=open_return.pow(2).ewm(alpha=.06,adjust=False).mean()*252
    return features

def macro_asof(macro,decision_dates):
    left=pd.DataFrame({"decision_date":pd.to_datetime(decision_dates)})
    right=macro.copy();right["available_date"]=pd.to_datetime(right.available_date);right["vintage_date"]=pd.to_datetime(right.vintage_date)
    joined=pd.merge_asof(left.sort_values("decision_date"),right.sort_values("available_date"),left_on="decision_date",right_on="available_date",direction="backward")
    valid=joined.available_date.notna()
    if (joined.loc[valid,"available_date"]>joined.loc[valid,"decision_date"]).any():raise AssertionError("Macro availability leakage")
    return joined.set_index("decision_date")

def make_panel(prices,macro,config,include_unlabelled=False):
    calendar=next(iter(prices.values())).index;h=config["horizon_sessions"];warmup=config["warmup_sessions"]
    positions=list(range(warmup,len(calendar),h));decisions=calendar[positions]
    macro_table=macro_asof(macro,decisions);rows=[]
    for asset,frame in prices.items():
        features=price_features(frame)
        for i in positions:
            date=calendar[i]
            row={"date":date,"asset":asset,"position":i,**features.loc[date].to_dict(),**macro_table.loc[date].to_dict()}
            if i+h+1<len(calendar):
                # Features at close t; buy at next open t+1; sell at open t+h+1.
                log_returns=np.diff(np.log(frame.adjusted_open.iloc[i+1:i+h+2].to_numpy()))
                row.update(target_variance=float(np.mean(log_returns**2)*252),label_end=calendar[i+h+1],entry_date=calendar[i+1],exit_date=calendar[i+h+1],asset_return=float(frame.adjusted_open.iloc[i+h+1]/frame.adjusted_open.iloc[i+1]-1))
            elif not include_unlabelled:continue
            else:row.update(target_variance=np.nan,label_end=pd.NaT,entry_date=pd.NaT,exit_date=pd.NaT,asset_return=np.nan)
            rows.append(row)
    panel=pd.DataFrame(rows).sort_values(["date","asset"]).reset_index(drop=True)
    panel=panel.dropna(subset=PRICE_FEATURES+MACRO_FEATURES)
    if (panel.vintage_date>=panel.date).any():raise AssertionError("Use strictly older macro vintages")
    panel["target_log_variance"]=np.log(panel.target_variance.clip(lower=1e-8))
    return panel

def eligible_training(panel,date):
    return panel.loc[(panel.date<date)&(panel.label_end<date)&panel.target_log_variance.notna()].copy()


def split_dates(panel,config):
    test=sorted(panel.loc[panel.date>=config["test_start"],"date"].unique())
    if not test:raise ValueError("No test dates")
    first_test=pd.Timestamp(test[0])
    validation=sorted(panel.loc[(panel.date>=config["validation_start"])&(panel.date<config["test_start"])&(panel.label_end<first_test),"date"].unique())
    if not validation:raise ValueError("No eligible validation dates")
    return [pd.Timestamp(d) for d in validation],[pd.Timestamp(d) for d in test]
