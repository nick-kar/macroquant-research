import io,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from macroquant.data import parse_alfred,DataError,market_download
from macroquant.features import price_features,make_panel,macro_asof,eligible_training,PRICE_FEATURES,MACRO_FEATURES
from macroquant.models import walk_forward_ml,garch_forecasts
from macroquant.metrics import qlike,block_bootstrap_mean,macro_tests,portfolio_statistics
from macroquant.backtest import risky_weights,backtest

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates=pd.bdate_range("2010-01-01",periods=900)
        rng=np.random.default_rng(3);cls.prices={}
        for asset in ["A","B"]:
            returns=rng.normal(.0001,.01,len(dates));price=100*np.exp(np.cumsum(returns))
            cls.prices[asset]=pd.DataFrame({"adjusted_open":price,"adjusted_close":price*np.exp(rng.normal(0,.002,len(dates)))},index=dates)
        vintages=pd.date_range("2009-12-31","2014-12-31",freq="ME")
        cls.macro=pd.DataFrame({"vintage_date":vintages,"available_date":vintages+pd.Timedelta(days=1),"unemployment":5.,"unemployment_change_3m":.1,"inflation_yoy":2.,"policy_rate":1.})
        cls.config={"assets":["A","B"],"horizon_sessions":5,"warmup_sessions":63,"refit_periods":13,"min_train_rows":30,"random_seed":42,"target_portfolio_volatility":.1,"max_asset_weight":.6,"bootstrap_repetitions":200,"bootstrap_block_periods":5}
        cls.panel=make_panel(cls.prices,cls.macro,cls.config)
    def test_price_features_cannot_see_future(self):
        frame=self.prices["A"].copy();date=frame.index[300];original=price_features(frame).loc[:date]
        frame.loc[frame.index>date,"adjusted_open"]*=10;frame.loc[frame.index>date,"adjusted_close"]*=5
        pd.testing.assert_frame_equal(original,price_features(frame).loc[:date])
    def test_features_are_price_scale_invariant(self):
        original=price_features(self.prices["A"]);scaled=price_features(self.prices["A"]*7)
        np.testing.assert_allclose(original.dropna(),scaled.dropna(),rtol=1e-8,atol=1e-10)
    def test_target_matches_next_open_holding_interval(self):
        row=self.panel.iloc[0];frame=self.prices[row.asset];i=int(row.position);h=self.config["horizon_sessions"]
        expected=np.diff(np.log(frame.adjusted_open.iloc[i+1:i+h+2])).var()+np.diff(np.log(frame.adjusted_open.iloc[i+1:i+h+2])).mean()**2
        self.assertAlmostEqual(row.target_variance,expected*252)
        self.assertEqual(row.entry_date,frame.index[i+1]);self.assertEqual(row.exit_date,frame.index[i+h+1])
    def test_labels_finish_after_decision(self):self.assertTrue((self.panel.label_end>self.panel.date).all())
    def test_target_periods_do_not_overlap(self):
        rows=self.panel.loc[self.panel.asset=="A"]
        self.assertTrue((rows.exit_date.iloc[:-1].to_numpy()==rows.entry_date.iloc[1:].to_numpy()).all())
    def test_macro_never_joins_future_vintage(self):
        frame=macro_asof(self.macro,[pd.Timestamp("2010-01-01"),pd.Timestamp("2010-01-31"),pd.Timestamp("2010-02-01")])
        self.assertEqual(frame.iloc[1].vintage_date,pd.Timestamp("2009-12-31"))
        self.assertEqual(frame.iloc[2].vintage_date,pd.Timestamp("2010-01-31"))
        self.assertTrue((frame.available_date<=frame.index).all())
    def test_macro_future_revisions_do_not_change_past_features(self):
        changed=self.macro.copy();cut=pd.Timestamp("2011-01-01")
        before=macro_asof(changed,[cut]);changed.loc[changed.available_date>cut,"unemployment"]=99
        pd.testing.assert_frame_equal(before,macro_asof(changed,[cut]))
    def test_missing_macro_stays_missing(self):
        result=macro_asof(self.macro,[pd.Timestamp("1990-01-01")]);self.assertTrue(pd.isna(result.unemployment.iloc[0]))
    def test_alfred_requires_confirmed_vintage_header(self):
        with self.assertRaises(DataError):parse_alfred(b"observation_date,UNRATE_20200101\n2009-01-01,5\n","UNRATE","2010-01-01")
    def test_alfred_rejects_future_observations(self):
        with self.assertRaises(DataError):parse_alfred(b"observation_date,UNRATE_20100101\n2011-01-01,5\n","UNRATE","2010-01-01")
    def test_alfred_blank_unpublished_values_are_not_filled(self):
        result=parse_alfred(b"observation_date,UNRATE_20100101\n2009-01-01,5\n2011-01-01,\n","UNRATE","2010-01-01")
        self.assertEqual(len(result),1)
    def test_training_excludes_unfinished_labels(self):
        date=self.panel.date.iloc[100];train=eligible_training(self.panel,date)
        self.assertTrue((train.label_end<date).all());self.assertFalse(train.empty)
        unfinished=self.panel.loc[(self.panel.date<date)&(self.panel.label_end>=date)]
        self.assertTrue(set(unfinished.index).isdisjoint(train.index))
    def test_ml_refit_audit_proves_label_purge(self):
        dates=sorted(self.panel.loc[self.panel.position>=400,"date"].unique())[:15]
        with threadpool_limits(limits=1):forecast,audit=walk_forward_ml(self.panel,dates,"ridge_macro",1.,self.config)
        self.assertEqual(len(audit),2);self.assertEqual(len(forecast),30)
        self.assertTrue(all(a["max_training_label_end"]<a["forecast_start"] for a in audit))
    def test_future_targets_do_not_change_an_earlier_forecast(self):
        date=self.panel.loc[self.panel.position>=400,"date"].iloc[0];modified=self.panel.copy()
        modified.loc[modified.label_end>=date,"target_log_variance"]=100
        with threadpool_limits(limits=1):
            original,_=walk_forward_ml(self.panel,[date],"ridge_price",1.,self.config)
            changed,_=walk_forward_ml(modified,[date],"ridge_price",1.,self.config)
        pd.testing.assert_frame_equal(original,changed)
    def test_garch_inputs_end_at_forecast_date(self):
        date=self.panel.loc[self.panel.position>=400,"date"].iloc[0]
        with threadpool_limits(limits=1):forecast,audit=garch_forecasts(self.prices,self.panel,[date],self.config)
        self.assertEqual(len(forecast),2);self.assertTrue((forecast.forecast_variance>0).all())
        for row in audit:
            if not row["fallback"]:self.assertEqual(row["last_input_date"],row["refit"])
    def test_risky_weights_respect_caps_and_no_leverage(self):
        w=risky_weights([.0001,.25],[[1,.2],[.2,1]],.1,.6)
        self.assertTrue((w>=0).all());self.assertTrue((w<=.6).all());self.assertLessEqual(w.sum(),1)
    def test_low_risk_target_increases_cash(self):
        corr=np.eye(2);low=risky_weights([.04,.04],corr,.02,.6);high=risky_weights([.04,.04],corr,.2,.6)
        self.assertLess(low.sum(),high.sum())
    def predictions(self):
        dates=sorted(self.panel.loc[self.panel.position>=400,"date"].unique())[:10]
        p=self.panel.loc[self.panel.date.isin(dates),["date","asset","rv20"]].rename(columns={"rv20":"forecast_variance"})
        return p
    def test_higher_costs_reduce_nav(self):
        p=self.predictions();zero,_,_=backtest(self.prices,self.panel,p,self.config,0);costly,_,_=backtest(self.prices,self.panel,p,self.config,25)
        self.assertLess(costly.nav.iloc[-1],zero.nav.iloc[-1]);self.assertTrue((costly.cost_fraction>=0).all())
    def test_initial_entry_and_liquidation_are_charged(self):
        p=self.predictions();daily,trades,stats=backtest(self.prices,self.panel,p,self.config,10)
        self.assertGreater(daily.cost_fraction.iloc[0],0);self.assertGreater(stats["liquidation_cost_fraction"],0)
    def test_buy_hold_has_no_intermediate_rebalance(self):
        _,trades,_=backtest(self.prices,self.panel,self.predictions(),self.config,10,"buy_hold")
        self.assertTrue(np.allclose(trades.turnover_l1.iloc[1:],0))
    def test_portfolio_records_daily_not_just_weekly_nav(self):
        daily,trades,_=backtest(self.prices,self.panel,self.predictions(),self.config,0)
        self.assertEqual(len(daily),len(trades)*5);self.assertFalse(daily.date.duplicated().any())
    def test_qlike_is_zero_for_perfect_variance(self):np.testing.assert_allclose(qlike([.01,.1],[.01,.1]),[0,0])
    def test_qlike_floor_handles_small_variance(self):self.assertTrue(np.isfinite(qlike([0],[0])).all())
    def test_bootstrap_is_reproducible(self):
        a=block_bootstrap_mean(np.arange(30),100,5,42);b=block_bootstrap_mean(np.arange(30),100,5,42)
        self.assertEqual(a,b)
    def test_macro_contrasts_preserve_pairing_and_adjust_pvalues(self):
        rows=[]
        for date in pd.date_range("2020-01-01",periods=30,freq="7D"):
            for family in ["ridge","hgb"]:
                for suffix,pred in [("price",.08),("macro",.04)]:
                    for asset in ["A","B"]:rows.append({"date":date,"asset":asset,"model":family+"_"+suffix,"target_variance":.04,"forecast_variance":pred})
        result=macro_tests(pd.DataFrame(rows),self.config)
        self.assertTrue((result.mean_difference<0).all());self.assertTrue((result.holm_adjusted_p>=result.p_two_sided).all())
    def test_no_synthetic_fallback_for_bad_price_response(self):
        with tempfile.TemporaryDirectory() as d:
            with patch("macroquant.data.get",return_value=b"<html>unavailable</html>"):
                with self.assertRaises(DataError):market_download("SPY","2005-01-01","2025-12-31",d)
    def test_validation_selection_uses_only_labels_available_before_test(self):
        from macroquant.features import split_dates
        c={**self.config,"validation_start":"2011-01-01","test_start":"2012-01-01"}
        validation,test=split_dates(self.panel,c)
        rows=self.panel.loc[self.panel.date.isin(validation)]
        self.assertLess(rows.label_end.max(),test[0])

    def test_paper_records_cannot_be_replaced_or_backdated(self):
        from macroquant.paper import append_forecast
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ledger.sqlite"
            append_forecast({"signal_date":"2026-01-02","value":1},path)
            with self.assertRaises(ValueError):append_forecast({"signal_date":"2026-01-02","value":2},path)
            with self.assertRaises(ValueError):append_forecast({"signal_date":"2026-01-01","value":2},path)
    def test_paper_records_form_a_hash_chain(self):
        from macroquant.paper import append_forecast
        from contextlib import closing
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ledger.sqlite"
            first=append_forecast({"signal_date":"2026-01-02","value":1},path)
            append_forecast({"signal_date":"2026-01-03","value":2},path)
            with closing(sqlite3.connect(path)) as db:
                previous=db.execute("SELECT previous_hash FROM forecasts WHERE signal_date='2026-01-03'").fetchone()[0]
            self.assertEqual(previous,first)

if __name__=="__main__":unittest.main()
