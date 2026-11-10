import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from arch import arch_model
from macroquant.features import PRICE_FEATURES,MACRO_FEATURES,eligible_training

MODEL_NAMES=["rv20","ewma","garch","ridge_price","ridge_macro","hgb_price","hgb_macro"]

def model_pipeline(name,parameter,seed):
    features=PRICE_FEATURES+(MACRO_FEATURES if name.endswith("macro") else [])
    transform=ColumnTransformer([("numeric",StandardScaler(),features),("asset",OneHotEncoder(handle_unknown="ignore",sparse_output=False),["asset"])])
    estimator=Ridge(alpha=parameter) if name.startswith("ridge") else HistGradientBoostingRegressor(max_leaf_nodes=int(parameter),max_iter=100,learning_rate=.05,min_samples_leaf=25,l2_regularization=1,random_state=seed,early_stopping=False)
    return Pipeline([("features",transform),("model",estimator)])

def walk_forward_ml(panel,dates,name,parameter,config):
    forecasts=[];audit=[];model=None
    for counter,date in enumerate(dates):
        current=panel.loc[panel.date==date]
        if model is None or counter%config["refit_periods"]==0:
            train=eligible_training(panel,date)
            if len(train)<config["min_train_rows"]:raise ValueError("Insufficient eligible training rows")
            model=model_pipeline(name,parameter,config["random_seed"]);model.fit(train,train.target_log_variance)
            audit.append({"model":name,"forecast_start":str(pd.Timestamp(date).date()),"max_training_label_end":str(train.label_end.max().date()),"training_rows":len(train),"parameter":parameter})
            if not train.label_end.max()<date:raise AssertionError("Training label leakage")
        pred=np.exp(np.clip(model.predict(current),-18,4))
        for (_,row),value in zip(current.iterrows(),pred):forecasts.append({"date":date,"asset":row.asset,"forecast_variance":float(max(value,1e-8)),"model":name})
    return pd.DataFrame(forecasts),audit

def garch_forecasts(prices,panel,dates,config):
    results=[];audit=[];h=config["horizon_sessions"]
    for asset,frame in prices.items():
        returns=np.log(frame.adjusted_open).diff()*100
        params=None;last_position=None;last_h=None;last_r=None
        for counter,date in enumerate(dates):
            position=frame.index.get_loc(date);fallback=False
            if params is None or counter%config["refit_periods"]==0:
                observed=returns.iloc[max(1,position-755):position+1].dropna()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit=arch_model(observed,mean="Zero",vol="GARCH",p=1,q=1,dist="normal",rescale=False).fit(disp="off",show_warning=False)
                    if fit.convergence_flag!=0:raise ValueError("GARCH nonconvergence")
                    params=fit.params;last_h=float(fit.conditional_volatility.iloc[-1]**2);last_r=float(observed.iloc[-1]);last_position=position
                    audit.append({"model":"garch","asset":asset,"refit":str(date.date()),"last_input_date":str(observed.index[-1].date()),"observations":len(observed),"fallback":False})
                except Exception:
                    params=None;fallback=True
                    audit.append({"model":"garch","asset":asset,"refit":str(date.date()),"fallback":True})
            if params is not None:
                omega=float(params["omega"]);alpha=float(params["alpha[1]"]);beta=float(params["beta[1]"])
                for j in range(last_position+1,position+1):
                    last_h=omega+alpha*last_r**2+beta*last_h;last_r=float(returns.iloc[j])
                last_position=position
                future=[omega+alpha*last_r**2+beta*last_h]
                for _ in range(h):future.append(omega+(alpha+beta)*future[-1])
                # First open-to-open step ends at entry; held returns are steps 2..h+1.
                variance=float(np.mean(future[1:])*252/10000)
            else:
                variance=float(panel.loc[(panel.date==date)&(panel.asset==asset),"ewma"].iloc[0])
            results.append({"date":date,"asset":asset,"model":"garch","forecast_variance":max(variance,1e-8)})
    return pd.DataFrame(results),audit
