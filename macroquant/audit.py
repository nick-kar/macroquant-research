"""Verify timing and accounting invariants in an executed research directory."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def audit(directory):
    directory=Path(directory);checks=[]
    summary=json.loads((directory/'summary.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    assert selection['test_used_for_selection'] is False
    checks.append('Model and hyperparameter selection records exclude test performance')
    panel=pd.read_csv(directory/'research_panel.csv',parse_dates=['date','vintage_date','available_date','label_end','entry_date','exit_date'])
    assert (panel.vintage_date<panel.date).all()
    assert (panel.available_date<=panel.date).all()
    assert (panel.entry_date>panel.date).all()
    assert (panel.label_end==panel.exit_date).all()
    assert (panel.exit_date>panel.entry_date).all()
    checks.append('Macro availability, feature timing and future holding intervals are consistent')
    training=pd.read_csv(directory/'training_audit.csv')
    fitted=training.dropna(subset=['forecast_start','max_training_label_end'])
    assert (pd.to_datetime(fitted.max_training_label_end)<pd.to_datetime(fitted.forecast_start)).all()
    checks.append('All learned-model refits use only completed earlier labels')
    test=pd.read_csv(directory/'test_predictions.csv',parse_dates=['date'])
    validation=pd.read_csv(directory/'validation_predictions.csv',parse_dates=['date'])
    assert validation.date.max()<test.date.min()
    assert pd.to_datetime(validation.label_end).max()<test.date.min()
    assert not test.duplicated(['date','asset','model']).any()
    assert len(test.model.unique())==7
    assert np.isfinite(test.forecast_variance).all() and (test.forecast_variance>0).all()
    assert (test.groupby(['date','model']).asset.nunique()==len(summary['assets'])).all()
    checks.append('Seven models, complete asset alignment, positive finite forecasts and chronological split')
    paths=pd.read_csv(directory/'portfolio_daily_paths.csv',parse_dates=['date'])
    for model,group in paths.groupby('model'):
        assert not group.date.duplicated().any()
        assert np.allclose(group.nav.to_numpy(),np.cumprod(1+group.net_return.to_numpy()))
        assert len(group)==summary['test_periods']*5
    checks.append('Daily portfolio paths compound recorded net returns over every held interval')
    portfolio=pd.read_csv(directory/'portfolio_metrics.csv')
    for model,group in portfolio.groupby('model'):
        values=group.sort_values('cost_bps').terminal_nav.to_numpy()
        assert (np.diff(values)<=1e-10).all()
    checks.append('Increasing proportional trading costs never improves terminal wealth')
    for asset in summary['assets']:
        rows=panel.loc[panel.asset==asset]
        assert (rows.exit_date.iloc[:-1].to_numpy()==rows.entry_date.iloc[1:].to_numpy()).all()
    checks.append('Five-session targets join without overlap or omitted holding intervals')
    result={'passed':True,'checks':checks,'scope':'Deterministic internal timing/accounting invariants. Not independent provider-data validation or proof of investment performance.'}
    (directory/'audit_results.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
if __name__=='__main__':main()
