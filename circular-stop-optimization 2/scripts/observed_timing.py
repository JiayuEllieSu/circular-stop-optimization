#!/usr/bin/env python3
"""Out-of-period timing evaluation on published Orangeline observations."""
from __future__ import annotations
import argparse
import json
import itertools
import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'observed'
OUT = ROOT / 'results' / 'observed'
MODELS = ('pooled_mean', 'pooled_median', 'site_mean', 'site_median')
LABELS = {'pooled_mean':'Pooled mean','pooled_median':'Pooled median',
          'site_mean':'Location mean','site_median':'Location median'}

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / 'orange_timing_long.csv', keep_default_na=True)
    if len(df) != 390 or df.duplicated(['quantity','site','collection_column']).any():
        raise ValueError('Unexpected table shape or duplicate cell keys.')
    if not df.loc[df.seconds.notna(),'seconds'].gt(0).all():
        raise ValueError('Observed durations must be positive; missing is not zero.')
    return df

def fitting_rows(df: pd.DataFrame, quantity: str) -> pd.DataFrame:
    x = df.loc[df.quantity.eq(quantity)].copy()
    if quantity == 'stopped':
        x = x.loc[~x.site.isin([1,20])].copy()
    return x

def fit(train: pd.DataFrame, model: str) -> tuple[pd.Series,float]:
    if model not in MODELS:
        raise ValueError(f'Unknown model: {model}')
    if not train.period.eq('train').all():
        raise ValueError('Refusing to fit on non-training observations.')
    observed = train.dropna(subset=['seconds'])
    if observed.empty:
        raise ValueError('No training observations.')
    statistic = 'median' if model.endswith('median') else 'mean'
    fallback = float(getattr(observed.seconds,statistic)())
    loc = getattr(observed.groupby('site').seconds,statistic)()
    if model.startswith('pooled_'):
        loc[:] = fallback
    return loc,fallback

def predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    params,fallback = fit(train,model)
    return test.site.map(params).fillna(fallback).to_numpy(float)

def score(errors: np.ndarray) -> dict:
    if len(errors) == 0:
        raise ValueError('Empty evaluation set.')
    return {'n_observations':int(len(errors)), 'mae_seconds':float(np.abs(errors).mean()),
            'rmse_seconds':float(np.sqrt(np.square(errors).mean())),
            'bias_pred_minus_observed_seconds':float(errors.mean())}

def transcription_audit(df: pd.DataFrame) -> pd.DataFrame:
    pub=pd.read_csv(DATA/'published_summary_checks.csv')
    rows=[]
    for row in pub.itertuples():
        vals=df.loc[df.quantity.eq(row.quantity)&df.site.eq(row.site),'seconds'].dropna()
        mean=float(vals.mean()) if len(vals) else np.nan
        sd=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
        rows.append({'quantity':row.quantity,'site':row.site,'n_values':len(vals),
                     'calculated_mean':mean,'published_mean':row.reported_mean_seconds,
                     'mean_difference':mean-row.reported_mean_seconds,
                     'calculated_sample_sd':sd,'published_sample_sd':row.reported_sample_sd_seconds,
                     'sd_difference':sd-row.reported_sample_sd_seconds})
    return pd.DataFrame(rows)

def run() -> dict:
    OUT.mkdir(parents=True,exist_ok=True)
    df=load(); metrics=[]; daily=[]; predictions=[]; fitted=[]; coverage=[]
    for quantity in ['travel','stopped']:
        x=fitting_rows(df,quantity)
        train=x[x.period.eq('train')].copy()
        test=x[x.period.eq('test')].dropna(subset=['seconds']).copy()
        support=set(train.dropna(subset=['seconds']).site)
        coverage.append({'quantity':quantity,'train_observed':int(train.seconds.notna().sum()),
                         'train_cells':len(train),'test_observed':len(test),
                         'test_cells':int(x.period.eq('test').sum()),
                         'common_support_observed':int(test.site.isin(support).sum()),
                         'test_days':int(test.source_date_label.nunique()),
                         'unseen_test_sites':sorted(set(test.site)-support)})
        for model in MODELS:
            loc,fallback=fit(train,model)
            for site in sorted(x.site.unique()):
                fitted.append({'quantity':quantity,'model':model,'site':int(site),
                               'prediction_seconds':float(loc.get(site,fallback)),
                               'fallback_used':site not in loc.index,
                               'training_observations':int((train.site.eq(site)&train.seconds.notna()).sum())})
            pred=predict(train,test,model)
            for row,yhat in zip(test.itertuples(),pred):
                predictions.append({'quantity':quantity,'model':model,'site':row.site,
                    'source_date_label':row.source_date_label,'collection_column':row.collection_column,
                    'observed_seconds':row.seconds,'prediction_seconds':float(yhat),
                    'error_seconds':float(yhat-row.seconds),'common_support':row.site in support})
            for subset in ['common_support','all_recorded']:
                take=test.site.isin(support).to_numpy() if subset=='common_support' else np.ones(len(test),bool)
                used=test.loc[take].copy(); used['error']=pred[take]-used.seconds.to_numpy()
                used['service_day']=pd.to_datetime(used.source_date_label).dt.strftime('%Y-%m-%d')
                rec={'quantity':quantity,'support':subset,'model':model,**score(used.error.to_numpy())}
                dm=[]
                for date,g in used.groupby('service_day',sort=True):
                    s=score(g.error.to_numpy()); dm.append(s['mae_seconds'])
                    daily.append({'quantity':quantity,'support':subset,'model':model,'date':date,**s})
                rec.update(day_macro_mae_seconds=float(np.mean(dm)),min_day_mae_seconds=float(min(dm)),max_day_mae_seconds=float(max(dm)))
                metrics.append(rec)
    met=pd.DataFrame(metrics); day=pd.DataFrame(daily); pred=pd.DataFrame(predictions)
    block_results=[]
    for quantity in ['travel','stopped']:
        d=day[(day.quantity==quantity)&(day.support=='common_support')].pivot(index='date',columns='model',values='mae_seconds')
        gain=(d.pooled_mean-d.site_mean).to_numpy()
        draws=np.array(list(itertools.product(range(len(gain)),repeat=len(gain))))
        boot=gain[draws].mean(axis=1)
        block_results.append({'quantity':quantity,'day_macro_mae_reduction_seconds':float(gain.mean()),
            'bootstrap_percentile_2_5_seconds':float(np.quantile(boot,.025)),
            'bootstrap_percentile_97_5_seconds':float(np.quantile(boot,.975)),
            'days_improved':int((gain>0).sum()),'held_out_days':len(gain),
            'bootstrap_resamples':len(boot),
            'scope':'Descriptive five-day paired bootstrap, conditional on fitted models; no training, transcription or counterfactual uncertainty.'})
    pd.DataFrame(block_results).to_csv(OUT/'day_block_bootstrap.csv',index=False)
    met.to_csv(OUT/'timing_metrics.csv',index=False)
    day.to_csv(OUT/'daily_metrics.csv',index=False)
    pred.to_csv(OUT/'held_out_predictions.csv',index=False)
    pd.DataFrame(fitted).to_csv(OUT/'training_fits.csv',index=False)
    audit=transcription_audit(df);audit.to_csv(OUT/'transcription_audit.csv',index=False)
    od=pd.read_csv(DATA/'orange_survey_od.csv').pivot(index='origin_table9_id',columns='destination_table9_id',values='count')
    pd.DataFrame({'table9_stop_id':od.index,'boardings':od.sum(axis=1),'alightings':od.sum(axis=0).to_numpy()}).to_csv(OUT/'survey_marginals.csv',index=False)
    report={'analysis':'Published-observation timing calibration, not stop-design validation',
            'coverage':coverage,'survey_count':int(od.to_numpy().sum()),'survey_nonzero_cells':int((od>0).to_numpy().sum()),
            'max_abs_published_mean_discrepancy_seconds':float(audit.mean_difference.abs().max()),
            'max_abs_published_sd_discrepancy_seconds':float(audit.sd_difference.abs().max()),
            'missing_treated_as_zero':False,'fits_use_test_values':False,
            'uncertainty':'Five distinct held-out days; report per-day errors, not independent-segment significance tests.',
            'crosswalk_or_pedestrian_network_verified':False,'counterfactual_savings_evaluated':False}
    (OUT/'analysis_summary.json').write_text(json.dumps(report,indent=2))
    print(met.to_string(index=False));print(json.dumps(report,indent=2))
    return report

class EmpiricalTests(unittest.TestCase):
    def setUp(self): self.df=load()
    def test_counts_and_missingness(self):
        self.assertEqual(len(self.df),390)
        self.assertTrue(self.df.loc[self.df.source_token.eq('-'),'seconds'].isna().all())
        self.assertEqual(int(self.df.loc[self.df.quantity.eq('travel')&self.df.period.eq('test'),'seconds'].notna().sum()),95)
    def test_chronology(self):
        dates=pd.to_datetime(self.df.source_date_label)
        self.assertLess(dates[self.df.period.eq('train')].max(),dates[self.df.period.eq('test')].min())
        self.assertEqual(dates[self.df.period.eq('test')].nunique(),5)
    def test_published_summary_agreement(self):
        a=transcription_audit(self.df)
        self.assertLess(a.mean_difference.abs().max(),.002)
        self.assertLess(a.sd_difference.abs().max(),.002)
    def test_terminals_and_header_not_silently_changed(self):
        d=self.df[self.df.quantity.eq('stopped')]
        np.testing.assert_allclose(d[d.site.eq(1)].seconds,d[d.site.eq(20)].seconds,equal_nan=True)
        self.assertEqual(d[(d.site==1)&(d.collection_column==1)].source_date_label.iloc[0],'10/18/2018')
        self.assertFalse(fitting_rows(self.df,'stopped').site.isin([1,20]).any())
    def test_fallback_no_january_leakage(self):
        for q in ['travel','stopped']:
            x=fitting_rows(self.df,q); tr=x[x.period=='train']; te=x[x.period=='test'].dropna(subset=['seconds'])
            perturbed=te.copy();perturbed['seconds']=1e9
            for m in MODELS:
                np.testing.assert_allclose(predict(tr,te,m),predict(tr,perturbed,m))
        with self.assertRaises(ValueError):fit(te,'site_mean')
    def test_od_marginals(self):
        d=pd.read_csv(DATA/'orange_survey_od.csv').pivot(index='origin_table9_id',columns='destination_table9_id',values='count')
        self.assertEqual(d.to_numpy().sum(),23); self.assertEqual(d.loc[11,6],1)
        np.testing.assert_array_equal(d.sum(0),[0,0,2,0,3,2,0,0,0,4,2,2,1,1,0,0,0,0,6])
        np.testing.assert_array_equal(d.sum(1),[12,0,0,0,1,5,0,1,0,1,1,2,0,0,0,0,0,0,0])
    def test_unseen_link_is_not_fitted_from_test(self):
        x=fitting_rows(self.df,'travel'); tr=x[x.period=='train'];te=x[(x.period=='test')&(x.site==12)]
        loc,fallback=fit(tr,'site_mean');self.assertNotIn(12,loc.index)
        np.testing.assert_allclose(predict(tr,te,'site_mean'),fallback)
    def test_simple_hand_calculation(self):
        tr=pd.DataFrame({'site':[1,1,2],'seconds':[10.,30.,50.],'period':['train']*3})
        te=pd.DataFrame({'site':[1,2,3]})
        np.testing.assert_allclose(predict(tr,te,'site_mean'),[20.,50.,30.])
        self.assertEqual(score(np.array([2.,-2.]))['mae_seconds'],2.)
        self.assertEqual(score(np.array([2.,-2.]))['rmse_seconds'],2.)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',action='store_true'); args=parser.parse_args()
    if args.test:
        unittest.main(argv=['empirical_validation.py'])
    else:run()
