#!/usr/bin/env python3
"""Plot benchmark and timing results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
R=ROOT/'results'
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})

def finish(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    plt.close(fig)

phase=pd.read_csv(R/'case_phases.csv');case=pd.read_csv(R/'case_methods.csv').set_index('method')
best=case.loc['exact_reassigned','train_minutes'];baseline=case.loc['reference','train_minutes']
fig,ax=plt.subplots(figsize=(6.5,3.5))
ax.plot(phase.phase,60*(phase.train_minutes-best),marker='o',label='Fixed-density quantile projection')
ax.axhline(60*(baseline-best),linestyle='--',label='Reference layout')
ax.axhline(0,linestyle=':',label='Exact training optimum')
ax.axvline(.5,linestyle='-.',alpha=.6,label='Original phase (0.5)')
ax.set(xlabel='Quantile phase',ylabel='Training excess cost (seconds per trip)',xlim=(-.02,.96))
ax.legend(loc='upper left',frameon=False);ax.grid(axis='y',alpha=.2)
finish(fig,'phase_effect')

land=pd.read_csv(R/'case_landscape.csv')
fig,ax=plt.subplots(figsize=(6.5,3.8))
ax.scatter(60*(land.frozen_train_minutes-land.frozen_train_minutes.min()),
           60*(land.reassigned_train_minutes-land.reassigned_train_minutes.min()),s=11,alpha=.45,label='Feasible layouts (508)')
for name,label,marker in [('reference','Frozen-flow optimum','s'),('fixed_single','Single-phase density','^'),('exact_reassigned','Reassignment optimum','*')]:
    row=case.loc[name]
    ax.scatter(60*(row.frozen_minutes-land.frozen_train_minutes.min()),60*(row.train_minutes-land.reassigned_train_minutes.min()),s=85,marker=marker,label=label)
ax.set(xlabel='Frozen-flow excess cost (seconds per trip)',ylabel='Reassigned excess cost (seconds per trip)')
ax.legend(frameon=False,loc='upper left');ax.grid(alpha=.15)
finish(fig,'surrogate_landscape')

summary=pd.read_csv(R/'suite_summary.csv').set_index('method')
methods=['fixed_single','fixed_phase_reassigned','frozen_exact','uniform_phase_reassigned','one_swap_phase','two_swap_phase','two_swap_without_density','exact_reassigned']
labels=['Single-phase fixed density','Phase-selected fixed density','Exact frozen-flow design','Phase-selected uniform density','One-swap + phase starts','Two-swap + phase starts','Two-swap without density','Exact reassignment (training)']
fig,ax=plt.subplots(figsize=(6.7,4.1))
y=np.arange(len(methods))
means=summary.loc[methods,'mean_population_regret_seconds'].values
p95=summary.loc[methods,'p95_population_regret_seconds'].values
ax.barh(y,means,label='Mean regret')
ax.scatter(p95,y,marker='|',s=130,label='95th percentile across fits')
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Population excess cost (seconds per trip)')
ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',alpha=.2)
finish(fig,'benchmark_regret')

fig,ax=plt.subplots(figsize=(6.4,3.6))
ns=[300,3000,30000];paths=[R/'sample_300',R,R/'sample_30000']
for name,label,marker in [('fixed_phase_reassigned','Phase-selected density','o'),('two_swap_phase','Two-swap + phase starts','s'),('exact_reassigned','Exact training optimizer','^')]:
    means=[pd.read_csv(p/'suite_summary.csv').set_index('method').loc[name,'mean_population_regret_seconds'] for p in paths]
    ax.plot(ns,means,marker=marker,label=label)
ax.set_xscale('log');ax.set_xticks(ns,[f'{n:,}' for n in ns]);ax.minorticks_off()
ax.set(xlabel='Training passenger trips',ylabel='Mean population excess cost (seconds per trip)')
ax.legend(frameon=False);ax.grid(alpha=.2)
finish(fig,'sample_size')

suite=pd.read_csv(R/'suite_results.csv')
selected=['reference','fixed_single','fixed_phase_reassigned','two_swap_phase']
labels2=['Reference','Single-phase density','Phase-selected density','Two-swap + phase starts']
fig,ax=plt.subplots(figsize=(6.4,3.6))
x=np.arange(4);width=.18
for j,(method,label) in enumerate(zip(selected,labels2)):
    values=suite[suite.method==method].groupby('demand').population_regret_seconds.mean().reindex(['uniform','local','hubs','commuter'])
    ax.bar(x+(j-1.5)*width,values,width,label=label)
ax.set_xticks(x,['Uniform','Short local','Two-hub','Commuter'])
ax.set_ylabel('Mean population excess cost (seconds per trip)')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
finish(fig,'demand_families')

observed=pd.read_csv(R/'observed/daily_metrics.csv')
for quantity in ('travel','stopped'):
    fig,ax=plt.subplots(figsize=(7.2,3.3))
    for model,marker,label in (('pooled_mean','o','Pooled mean'),('site_mean','s','Location mean')):
        v=observed[(observed.quantity==quantity)&(observed.support=='common_support')&(observed.model==model)]
        ax.plot(range(len(v)),v.mae_seconds,marker=marker,label=label)
    ax.set_xticks(range(5),['Jan 7','Jan 8','Jan 9','Jan 10','Jan 11'])
    ax.set_ylabel('Mean absolute error (seconds)')
    ax.set_xlabel('Held-out collection day, 2019')
    ax.legend(frameon=False)
    ax.grid(axis='y',alpha=.25)
    finish(fig,f'observed_{quantity}_daily')
