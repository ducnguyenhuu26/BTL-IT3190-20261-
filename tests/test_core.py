import unittest
import numpy as np
import pandas as pd
from src.data import chronological_split
from src.features import make_features
from src.jump_filter import JumpAwareFilter
from src.levy_process import LevyOU
from src.evaluate import online_adapter_backtest
from src.baselines import EMAAdapter
from src.robust_jump_filter import RobustJumpFilter

class CoreTests(unittest.TestCase):
    def test_shifted_features_do_not_read_current_target(self):
        df=pd.DataFrame({"date":pd.date_range("2020-01-01",periods=10),"store_nbr":1,"family":"A","sales":np.arange(10,dtype=float)})
        a=make_features(df); changed=df.copy(); changed.loc[5,"sales"]=9999; b=make_features(changed)
        self.assertEqual(a.loc[5,"sales_lag1"],b.loc[5,"sales_lag1"])
        self.assertEqual(a.loc[5,"sales_roll_mean_7"],b.loc[5,"sales_roll_mean_7"])
        self.assertNotEqual(a.loc[6,"sales_lag1"],b.loc[6,"sales_lag1"])

    def test_missing_dates_remain_missing_in_calendar_lags(self):
        df=pd.DataFrame({"date":pd.to_datetime(["2020-01-01","2020-01-03","2020-01-08"]),"store_nbr":1,"family":"A","sales":[1.,3.,8.]})
        f=make_features(df)
        self.assertTrue(np.isnan(f.loc[1,"sales_lag1"]))
        self.assertEqual(f.loc[2,"sales_lag7"],1.)
        self.assertEqual(f.loc[2,"sales_roll_mean_7"],2.)

    def test_oil_feature_excludes_same_day_publication(self):
        df=pd.DataFrame({"date":pd.to_datetime(["2020-01-01","2020-01-02"]),"store_nbr":1,"family":"A","sales":[1.,2.]})
        oil=pd.DataFrame({"date":pd.to_datetime(["2020-01-01","2020-01-02"]),"dcoilwtico":[10.,20.]})
        f=make_features(df,{"oil":oil})
        self.assertTrue(np.isnan(f.loc[0,"oil_price_lag1"]))
        self.assertEqual(f.loc[1,"oil_price_lag1"],10.)

    def test_chronological_split_disjoint(self):
        df=pd.DataFrame({"date":pd.date_range("2015-12-30",periods=100),"sales":1})
        s=chronological_split(df)
        self.assertLess(s.train.date.max(),s.validation.date.min()); self.assertLess(s.validation.date.max(),s.test.date.min())

    def test_p_zero_matches_kalman_and_handles_extreme_residual(self):
        p=LevyOU(rho=.9,q=.03,p=0,q_jump=.4); f=JumpAwareFilter(p,.2); k=JumpAwareFilter(LevyOU(rho=.9,q=.03,p=0,q_jump=0),.2)
        for y in [0.1,-.4,1e100]:
            x=f.update("a",y); z=k.update("a",y)
            self.assertAlmostEqual(x["corrected_mean"],z["corrected_mean"])
            self.assertAlmostEqual(x["variance"],z["variance"])
            self.assertEqual(x["jump_probability"],0)
            self.assertTrue(np.isfinite(x["corrected_mean"]))

    def test_qjump_zero_and_new_series_are_finite(self):
        f=JumpAwareFilter(LevyOU(p=.1,q_jump=0),.1)
        self.assertEqual(f.prior("new")[0],0)
        self.assertTrue(np.isfinite(f.update("new",1e6)["variance"]))

    def test_same_day_forecasts_precede_updates(self):
        dates=pd.to_datetime(["2020-01-01","2020-01-01","2020-01-02"])
        frame=pd.DataFrame({"date":dates,"store_nbr":[1,2,1],"family":["A","A","A"],"sales":[2.,3.,4.]})
        base=np.log1p([2.,3.,4.])
        a=online_adapter_backtest(frame,base,EMAAdapter(.5),"ema")
        changed=frame.copy(); changed.loc[changed.date >= pd.Timestamp("2020-01-01"),"sales"] *= 1000
        b=online_adapter_backtest(changed,base,EMAAdapter(.5),"ema")
        np.testing.assert_array_equal(a.loc[:1,"prediction"],b.loc[:1,"prediction"])

    def test_qjump_zero_matches_kalman_even_when_p_positive(self):
        jump=JumpAwareFilter(LevyOU(rho=.93,q=.02,p=.4,q_jump=0),.15,prior_variance=.7)
        kalman=JumpAwareFilter(LevyOU(rho=.93,q=.02,p=0,q_jump=0),.15,prior_variance=.7)
        for y in [.4,-.7,1.2]:
            a=jump.update("s",y); b=kalman.update("s",y)
            self.assertAlmostEqual(a["corrected_mean"],b["corrected_mean"],places=12)
            self.assertAlmostEqual(a["variance"],b["variance"],places=12)
            self.assertGreater(a["variance"],0)

    def test_large_residual_with_active_jump_stays_finite(self):
        f=JumpAwareFilter(LevyOU(rho=.9,q=.01,p=.03,q_jump=.8),.2)
        x=f.update("active",1e100)
        self.assertTrue(np.isfinite(x["corrected_mean"]))
        self.assertTrue(np.isfinite(x["variance"]))
        self.assertGreater(x["variance"],0)
        self.assertTrue(np.isfinite(x["jump_probability"]))

    def test_three_branch_zero_probabilities_match_kalman_stepwise(self):
        robust=RobustJumpFilter(LevyOU(mu=.1,rho=.93,q=.02,p=0,q_jump=.7),.15,p_outlier=0,
                                prior_mean=.1,prior_variance=.7)
        kalman=JumpAwareFilter(LevyOU(mu=.1,rho=.93,q=.02,p=0,q_jump=0),.15,
                               prior_mean=.1,prior_variance=.7)
        for y in [.4,-.7,1.2]:
            np.testing.assert_allclose(robust.predict("s"),kalman.predict("s"),rtol=0,atol=1e-14)
            a=robust.update("s",y); b=kalman.update("s",y)
            np.testing.assert_allclose([a["corrected_mean"],a["variance"]],
                                       [b["corrected_mean"],b["variance"]],rtol=1e-12,atol=1e-14)
            self.assertEqual(a["jump_probability"],0)
            self.assertEqual(a["outlier_probability"],0)

    def test_three_branch_without_outliers_matches_two_branch(self):
        process=LevyOU(mu=-.05,rho=.91,q=.01,p=.03,q_jump=.8)
        three=RobustJumpFilter(process,.2,p_outlier=0,prior_mean=0,prior_variance=.9)
        two=JumpAwareFilter(process,.2,prior_mean=0,prior_variance=.9)
        for y in [.2,1.7,-.4,4.2]:
            a=three.update("s",y); b=two.update("s",y)
            np.testing.assert_allclose([a["prior_mean"],a["corrected_mean"],a["variance"],a["jump_probability"]],
                                       [b["prior_mean"],b["corrected_mean"],b["variance"],b["jump_probability"]],
                                       rtol=1e-11,atol=1e-13)
            self.assertEqual(a["outlier_probability"],0)

    def test_three_branch_weights_and_extreme_residual_are_stable(self):
        filt=RobustJumpFilter(LevyOU(rho=.9,q=.01,p=.02,q_jump=.4),.2,
                              p_outlier=.01,outlier_variance=5)
        state=filt.update("s",1e100)
        self.assertAlmostEqual(state["ordinary_probability"]+state["jump_probability"]+
                               state["outlier_probability"],1.,places=12)
        self.assertTrue(np.isfinite(state["corrected_mean"]))
        self.assertTrue(np.isfinite(state["variance"]))
        self.assertGreater(state["variance"],0)

    def test_isolated_outlier_prefers_observation_branch_and_limits_state_update(self):
        process=LevyOU(rho=.9,q=0,p=.005,q_jump=.1)
        robust=RobustJumpFilter(process,.1,p_outlier=.05,outlier_variance=10,
                                prior_variance=.01)
        ordinary=JumpAwareFilter(process,.1,prior_variance=.01)
        robust_result=robust.update("s",4.)
        ordinary_result=ordinary.update("s",4.)
        self.assertGreater(robust_result["outlier_probability"],robust_result["jump_probability"])
        self.assertLess(abs(robust_result["corrected_mean"]),abs(ordinary_result["corrected_mean"]))

    def test_persistent_jump_state_is_carried_forward(self):
        filt=RobustJumpFilter(LevyOU(rho=.95,q=0,p=.2,q_jump=.8),.05,
                              p_outlier=.001,outlier_variance=5,prior_variance=.01)
        first=filt.update("s",3.)
        next_prior=filt.predict("s")
        self.assertGreater(first["jump_probability"],first["outlier_probability"])
        self.assertGreater(next_prior,0.1)
        second=filt.update("s",3.)
        self.assertGreater(filt.predict("s"),0.1)

    def test_three_branch_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            RobustJumpFilter(LevyOU(p=.8),.2,p_outlier=.2,outlier_variance=1)
        with self.assertRaises(ValueError):
            RobustJumpFilter(LevyOU(),.2,p_outlier=.1,outlier_variance=.2)

    def test_vectorized_same_day_updates_match_independent_scalar_filters(self):
        process=LevyOU(rho=.92,q=.01,p=.025,q_jump=.6)
        batch=RobustJumpFilter(process,.2,p_outlier=.01,outlier_variance=4)
        scalar={key:RobustJumpFilter(process,.2,p_outlier=.01,outlier_variance=4) for key in ("a","b","c")}
        keys=["a","b","c"]; residuals=np.array([.2,2.1,-.8])
        prior, _=batch.prior_many(keys)
        expected_prior=np.array([scalar[key].predict(key) for key in keys])
        np.testing.assert_allclose(prior,expected_prior,rtol=0,atol=1e-14)
        got=batch.update_many(keys,residuals)
        expected=[scalar[key].update(key,float(value)) for key,value in zip(keys,residuals)]
        for name in ("corrected_mean","variance","jump_probability","outlier_probability"):
            np.testing.assert_allclose(got[name],[row[name] for row in expected],rtol=1e-11,atol=1e-13)

    def test_vectorized_two_branch_update_matches_scalar(self):
        process=LevyOU(rho=.9,q=.01,p=.02,q_jump=.5)
        batch=JumpAwareFilter(process,.2)
        scalar={key:JumpAwareFilter(process,.2) for key in ("a","b")}
        keys=["a","b"]; residuals=np.array([1.3,-.4])
        got=batch.update_many(keys,residuals)
        expected=[scalar[key].update(key,float(value)) for key,value in zip(keys,residuals)]
        for name in ("corrected_mean","variance","jump_probability"):
            np.testing.assert_allclose(got[name],[row[name] for row in expected],rtol=1e-11,atol=1e-13)

    def test_generator_reproducible(self):
        proc=LevyOU(p=.2,q=.01,q_jump=.4)
        x,j=proc.sample(100,7); y,k=proc.sample(100,7)
        np.testing.assert_array_equal(x,y); np.testing.assert_array_equal(j,k)

if __name__=="__main__": unittest.main()
