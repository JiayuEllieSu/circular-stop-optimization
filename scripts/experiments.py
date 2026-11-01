#!/usr/bin/env python3
"""Synthetic experiments for constrained circular stop placement."""
from __future__ import annotations
import argparse
from dataclasses import dataclass, replace
import itertools
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import unittest
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
import stop_design as core

EPS = 1e-9

@dataclass(frozen=True)
class Direction:
    sign: int
    fleet: int
    motion: float = 18.0
    dwell_scale: float = 1.0
    def __post_init__(self):
        if self.sign not in (-1,1) or self.fleet < 1 or int(self.fleet) != self.fleet:
            raise ValueError('Direction must be +/-1 with a positive integer fleet.')
        if not np.isfinite(self.motion) or self.motion <= 0 or not np.isfinite(self.dwell_scale) or self.dwell_scale <= 0:
            raise ValueError('Motion time and dwell scale must be positive and finite.')

SERVICES = {
    'symmetric': (Direction(1,3), Direction(-1,3)),
    'clockwise': (Direction(1,3),),
    'unequal': (Direction(1,4), Direction(-1,2)),
}

def bounded_density(A, B, widths, budget, lower, upper):
    """Solve the fixed-budget density problem with box constraints."""
    A,B,widths,lower,upper = [np.asarray(v,dtype=float) for v in (A,B,widths,lower,upper)]
    if A.ndim != 1 or any(v.shape != A.shape for v in (B,widths,lower,upper)):
        raise ValueError('All input arrays must have the same one-dimensional shape.')
    if not all(np.all(np.isfinite(v)) for v in (A,B,widths,lower,upper)) or not np.isfinite(budget):
        raise ValueError('Finite inputs required.')
    if np.any(A<=0) or np.any(B<0) or np.any(widths<=0) or np.any(lower<=0) or np.any(upper<lower):
        raise ValueError('Require A>0, B>=0, widths>0, and 0<lower<=upper.')
    lo_mass,hi_mass = float(widths@lower),float(widths@upper)
    if budget < lo_mass-EPS or budget > hi_mass+EPS:
        raise ValueError('Infeasible density budget.')
    def at(lam):
        c=B+lam
        n=upper.copy()
        mask=c>0
        n[mask]=np.clip(np.sqrt(A[mask]/c[mask]),lower[mask],upper[mask])
        return n
    a=-float(B.max())-1.
    b=float(np.max(A/lower**2))+1.
    if abs(budget-lo_mass)<1e-12:
        return lower.copy(),b
    if abs(budget-hi_mass)<1e-12:
        return upper.copy(),a
    lam=brentq(lambda t:float(widths@at(t))-budget,a,b,xtol=1e-12)
    n=at(lam)
    if abs(float(widths@n)-budget)>1e-8*max(1,budget):
        raise RuntimeError('Bounded multiplier solve failed the mass check.')
    return n,float(lam)


def route_assignment(ins, od, directions):
    """Assign fixed directional paths by motion time."""
    p=core.probabilities(od)
    delta=ins.zones[None,:]-ins.zones[:,None]
    length=np.stack([(d.sign*delta)%1 for d in directions])
    motion=np.stack([d.motion*length[i] for i,d in enumerate(directions)])
    ties=np.isclose(motion,motion.min(axis=0),atol=1e-12,rtol=0)
    pi=ties/ties.sum(axis=0)
    return p,pi,length


def frozen_coefficients(ins, od, directions, points=None, dwell=None):
    points=ins.candidates if points is None else np.asarray(points)
    dwell=ins.dwell if dwell is None else np.asarray(dwell)
    p,pi,length=route_assignment(ins,od,directions)
    B=np.zeros(len(points)); const=0.
    flows=[]; shares=[]
    for d_idx,d in enumerate(directions):
        pd=p*pi[d_idx]
        alpha=float(pd.sum()); shares.append(alpha)
        displacement=(d.sign*(points[:,None]-ins.zones[None,:]))%1
        crossed=(displacement[:,:,None]>EPS)&(displacement[:,:,None]<length[d_idx][None,:,:]-EPS)
        q=np.sum(crossed*pd[None,:,:],axis=(1,2)); flows.append(q)
        B+=dwell*d.dwell_scale*(alpha/(2*d.fleet)+q)
        const+=float(np.sum(pd*d.motion*length[d_idx]))+alpha*d.motion/(2*d.fleet)
    endpoint=p.sum(axis=0)+p.sum(axis=1)
    walk=ins.walking_minutes_per_loop*core.circular_distance(ins.zones[:,None],points[None,:])
    return endpoint,B,walk,const,np.array(flows),np.array(shares)


def journey_costs(ins, stops, directions=SERVICES['symmetric'], walk_scale=1., dwell_scale=1., allow_walk=True):
    """Evaluate the best passenger journey for each OD pair."""
    s=np.asarray(sorted(stops),int); u=ins.candidates[s]
    k=len(s); ride=np.full((k,k),np.inf)
    for d in directions:
        mu=ins.dwell[s]*dwell_scale*d.dwell_scale
        arc=(d.sign*(u[None,:]-u[:,None]))%1
        forward=(d.sign*(u[None,:]-u[:,None]))%1
        interior=(forward[:,None,:]>EPS)&(forward[:,None,:]<arc[:,:,None]-EPS)
        v=d.motion*arc+np.sum(interior*mu[None,None,:],axis=2)+.5*(mu[:,None]+mu[None,:])
        cycle=d.motion+float(mu.sum())
        v+=cycle/(2*d.fleet)
        np.fill_diagonal(v,np.inf)
        ride=np.minimum(ride,v)
    walking=ins.walking_minutes_per_loop*walk_scale
    access=walking*core.circular_distance(ins.zones[:,None],u[None,:])
    via=(access[:,:,None]+ride[None,:,:]).min(axis=1)
    bus=(via[:,None,:]+access[None,:,:]).min(axis=2)
    direct=walking*core.circular_distance(ins.zones[:,None],ins.zones[None,:])
    return (np.minimum(bus,direct) if allow_walk else bus),bus<direct-EPS


def enumerate_feasible(ins, limit=100000):
    optional=[j for j in range(len(ins.candidates)) if j not in ins.mandatory and j not in ins.forbidden]
    count=ins.budget-len(ins.mandatory)
    if count<0 or count>len(optional):
        raise ValueError('Infeasible cardinality.')
    total=math.comb(len(optional),count)
    if total>limit:
        raise ValueError(f'Enumeration guard: {total} subsets exceeds {limit}. No global certificate.')
    layouts=[tuple(sorted((*ins.mandatory,*extra))) for extra in itertools.combinations(optional,count)
             if core.feasible(ins,tuple(sorted((*ins.mandatory,*extra))))]
    if not layouts:
        raise ValueError('No feasible layout.')
    return np.array(layouts,dtype=int),total


def density_profiles(ins,od,directions,bins=128):
    p=core.probabilities(od); endpoint=p.sum(axis=0)+p.sum(axis=1)
    masses,edges=np.histogram(ins.zones,bins=bins,range=(0,1),weights=endpoint)
    masses=.99*gaussian_filter1d(masses,2.,mode='wrap')+.01*(2/bins)
    u=(edges[1:]+edges[:-1])/2
    mu=np.interp(u,np.r_[ins.candidates[-1]-1,ins.candidates,ins.candidates[0]+1],
                 np.r_[ins.dwell[-1],ins.dwell,ins.dwell[0]])
    _,B,_,_,_,_=frozen_coefficients(ins,p,directions,u,mu)
    A=ins.walking_minutes_per_loop*masses*bins/4
    widths=np.full(bins,1/bins)
    free=np.sqrt(A/B); rescaled=free*ins.budget/float(widths@free)
    fixed,lam=core.fixed_budget_density(A,B,widths,ins.budget)
    gap=core.continuum_cost(A,B,widths,rescaled)-core.continuum_cost(A,B,widths,fixed)
    identity=float(widths@((B+lam)*(rescaled-fixed)**2/rescaled))
    info={'lambda':lam,'mass':float(widths@fixed),'free_mass':float(widths@free),
          'continuous_improvement_minutes':gap,'identity_error':abs(gap-identity)}
    return fixed,rescaled,edges,info


def projection_indices(ins,layouts,density,edges,phases):
    """Project density quantiles onto feasible candidate layouts."""
    cdf=np.r_[0.,np.cumsum(density*np.diff(edges))]; cdf/=cdf[-1]
    projected=[]; costs=[]
    for phase in phases:
        targets=np.interp((np.arange(ins.budget)+phase)/ins.budget,cdf,edges)
        cost=core.circular_distance(targets[:,None],ins.candidates[None,:])
        values=[]
        for s in layouts:
            mat=cost[:,s]
            rr,cc=linear_sum_assignment(mat)
            values.append(float(mat[rr,cc].sum()))
        idx=int(np.argmin(values));projected.append(idx);costs.append(values[idx])
    return np.array(projected),np.array(costs)


def incidence_and_neighbors(layouts,m):
    incidence=np.zeros((len(layouts),m),int)
    for i,s in enumerate(layouts): incidence[i,s]=1
    exchanges=layouts.shape[1]-incidence@incidence.T
    near1=[np.flatnonzero(exchanges[i]<=1) for i in range(len(layouts))]
    near2=[np.flatnonzero(exchanges[i]<=2) for i in range(len(layouts))]
    components,labels=connected_components(csr_matrix(exchanges==1),directed=False)
    return incidence,near1,near2,components,labels


def descend(values,start,neighbors):
    curr=int(start);steps=0
    while True:
        candidates=neighbors[curr]
        best=int(candidates[np.argmin(values[candidates])])
        if values[best]>=values[curr]-1e-10:
            return curr,steps
        curr=best;steps+=1
        if steps>len(values): raise RuntimeError('Descent cycle.')


def candidate_metrics(ins,layouts,od,directions,incidence=None):
    a,B,c,const,_,_=frozen_coefficients(ins,od,directions)
    access=np.array([float(a@c[:,s].min(axis=1)) for s in layouts])
    penalty=np.array([float(B[s].sum()) for s in layouts]) if incidence is None else incidence@B
    return access,access+penalty+const


def select_methods(ins,layouts,train,od,directions,neighbors1,neighbors2,precomputed_uniform=None):
    fixed,rescaled,edges,info=density_profiles(ins,od,directions)
    phases=np.arange(16)/16
    fidx,_=projection_indices(ins,layouts,fixed,edges,phases)
    ridx,_=projection_indices(ins,layouts,rescaled,edges,[.5])
    if precomputed_uniform is None:
        uidx,_=projection_indices(ins,layouts,np.full(128,ins.budget),edges,phases)
    else: uidx=precomputed_uniform
    access,G=candidate_metrics(ins,layouts,od,directions)
    ref=int(np.flatnonzero(np.all(layouts==ins.reference,axis=1))[0])
    best=lambda arr,score:int(arr[np.argmin(score[arr])])
    methods={'reference':ref,'rescaled_single':int(ridx[0]),'fixed_single':int(fidx[8]),
             'access_exact':int(np.argmin(access)),'frozen_exact':int(np.argmin(G)),
             'fixed_phase_frozen':best(fidx,G),'fixed_phase_reassigned':best(fidx,train),
             'uniform_phase_reassigned':best(uidx,train)}
    original_starts=[methods[n] for n in ('reference','rescaled_single','fixed_single','access_exact','frozen_exact')]
    base=[descend(train,i,neighbors1)[0] for i in set(original_starts)]
    methods['one_swap_original']=best(np.array(base),train)
    enhanced_starts=original_starts+[methods['fixed_phase_reassigned'],methods['uniform_phase_reassigned']]
    one=[descend(train,i,neighbors1)[0] for i in set(enhanced_starts)]
    methods['one_swap_phase']=best(np.array(one),train)
    two=[descend(train,i,neighbors2)[0] for i in set(enhanced_starts+one)]
    methods['two_swap_phase']=best(np.array(two),train)
    nonca=[methods[n] for n in ('reference','access_exact','frozen_exact','uniform_phase_reassigned')]
    nonca_one=[descend(train,i,neighbors1)[0] for i in set(nonca)]
    nonca_two=[descend(train,i,neighbors2)[0] for i in set(nonca+nonca_one)]
    methods['two_swap_without_density']=best(np.array(nonca_two),train)
    start_count=len(set(enhanced_starts))
    random_starts=set(nonca)
    seed=int.from_bytes(hashlib.sha256(np.asarray(od,dtype=np.int64).tobytes()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    available=np.array([i for i in range(len(layouts)) if i not in random_starts])
    needed=max(0,start_count-len(random_starts))
    if needed: random_starts.update(rng.choice(available,size=needed,replace=False).tolist())
    random_one=[descend(train,i,neighbors1)[0] for i in random_starts]
    random_two=[descend(train,i,neighbors2)[0] for i in random_starts | set(random_one)]
    methods['two_swap_matched_random']=best(np.array(random_two),train)
    methods['exact_reassigned']=int(np.argmin(train))
    e=train-G
    info.update({'unique_density_starts':start_count,'unique_matched_random_starts':len(random_starts),'phase_layouts':int(len(np.unique(fidx))),
                 'uniform_phase_layouts':int(len(np.unique(uidx))),
                 'surrogate_error_oscillation_minutes':float(np.ptp(e)),
                 'frozen_reassigned_spearman':float(spearmanr(G,train).statistic),
                 'quantile_single_frozen_gap_minutes':float(G[methods['fixed_single']]-G.min()),
                 'best_phase_train_gain_seconds':60*float(train[methods['fixed_single']]-train[methods['fixed_phase_reassigned']]),
                 'one_swap_train_gap_seconds':60*float(train[methods['one_swap_original']]-train.min()),
                 'two_swap_train_gap_seconds':60*float(train[methods['two_swap_phase']]-train.min())})
    for i in methods.values():
        lhs=train[i]-train.min(); rhs=G[i]-G.min()+np.ptp(e)
        if lhs>rhs+1e-8: raise RuntimeError('Surrogate-transfer bound failed.')
    return methods,info,G,fidx


def population(zones,kind):
    dist=core.circular_distance(zones[:,None],zones[None,:])
    def bump(c,s):return np.exp(-.5*(core.circular_distance(zones,c)/s)**2)
    if kind=='uniform': joint=np.ones_like(dist)
    elif kind=='local':joint=np.exp(-dist/.06)
    elif kind=='hubs':
        a=.15+2.5*bump(.13,.065)+1.8*bump(.59,.075)
        b=.15+2.2*bump(.34,.065)+1.7*bump(.85,.065)
        joint=.7*core.probabilities(np.outer(a,b))+.3*core.probabilities(np.exp(-dist/.12))
    elif kind=='commuter':
        a=.25+2*bump(.18,.08)+1.3*bump(.78,.06)
        b=.03+3*bump(.48,.045)
        joint=.65*core.probabilities(np.outer(a,b))+.25*core.probabilities(np.outer(b,a))+.1*core.probabilities(np.exp(-dist/.1))
    else: raise ValueError(f'Unknown demand family: {kind}')
    np.fill_diagonal(joint,0)
    return core.probabilities(joint)


def run_case(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ins=core.synthetic_instance();dirs=SERVICES['symmetric']
    layouts,total=enumerate_feasible(ins)
    incidence,n1,n2,components,labels=incidence_and_neighbors(layouts,len(ins.candidates))
    costs=[];bus=[]
    for s in layouts:
        cc,bb=journey_costs(ins,s,dirs);costs.append(cc.ravel());bus.append(bb.ravel())
    costs=np.array(costs);bus=np.array(bus)
    ptrain=core.probabilities(ins.train_counts).ravel();ptest=core.probabilities(ins.test_counts).ravel()
    train=costs@ptrain;test=costs@ptest
    methods,info,G,phase_indices=select_methods(ins,layouts,train,ins.train_counts,dirs,n1,n2)
    ref=methods['reference'];rng=np.random.default_rng(121)
    draws=rng.multinomial(int(ins.test_counts.sum()),ptest,size=1000)/ins.test_counts.sum()
    rows=[]
    for name,i in methods.items():
        diff=costs[ref]-costs[i];cis=np.quantile(draws@diff,[.025,.975])*60
        rows.append({'method':name,'stops':';'.join(map(str,layouts[i])),
          'train_minutes':float(train[i]),'test_minutes':float(test[i]),
          'saving_seconds':60*float(test[ref]-test[i]),'ci_low_seconds':float(cis[0]),'ci_high_seconds':float(cis[1]),
          'train_regret_seconds':60*float(train[i]-train.min()),'test_bus_share':float(bus[i]@ptest),
          'frozen_minutes':float(G[i]),'layout_index':i})
    core.save_csv(out/'case_methods.csv',rows)
    core.save_csv(out/'case_landscape.csv',[{'index':i,'stops':';'.join(map(str,s)),
         'frozen_train_minutes':float(G[i]),'reassigned_train_minutes':float(train[i]),
         'reassigned_test_minutes':float(test[i]),'one_swap_component':int(labels[i])} for i,s in enumerate(layouts)])
    core.save_csv(out/'case_phases.csv',[{'phase':j/16,'layout_index':int(i),'train_minutes':float(train[i]),'test_minutes':float(test[i]),'stops':';'.join(map(str,layouts[i]))} for j,i in enumerate(phase_indices)])
    scenario_train=[];scenario_test=[];scenarios=[]
    for w in (.8,1.,1.2):
        for d in (.5,1.,1.5):
            c=np.array([journey_costs(ins,s,dirs,w,d)[0].ravel() for s in layouts])
            scenario_train.append(c@ptrain);scenario_test.append(c@ptest);scenarios.append((w,d))
    ST=np.array(scenario_train); SE=np.array(scenario_test)
    regrets=ST-ST.min(axis=1,keepdims=True)
    robust=int(np.argmin(regrets.max(axis=0)))
    nominal=methods['exact_reassigned']
    robust_rows=[]
    for i,(w,d) in enumerate(scenarios):
        for name,j in [('reference',ref),('nominal_exact',nominal),('minimax_regret',robust)]:
            robust_rows.append({'walking_scale':w,'dwell_scale':d,'method':name,'stops':';'.join(map(str,layouts[j])),
                'train_regret_seconds':60*float(ST[i,j]-ST[i].min()),
                'test_regret_seconds':60*float(SE[i,j]-SE[i].min()),
                'test_saving_seconds':60*float(SE[i,ref]-SE[i,j])})
    core.save_csv(out/'case_robustness.csv',robust_rows)
    info.update({'candidate_subsets':total,'feasible_layouts':len(layouts),'one_swap_components':int(components),
                 'minimax_regret_stops':layouts[robust].tolist(),'nominal_stops':layouts[nominal].tolist(),
                 'nominal_max_training_scenario_regret_seconds':60*float(regrets[:,nominal].max()),
                 'robust_max_training_scenario_regret_seconds':60*float(regrets[:,robust].max()),
                 'bootstrap_replicates':1000,'original_test_size':int(ins.test_counts.sum())})
    (out/'case_metadata.json').write_text(json.dumps(info,indent=2))
    (out/'case_instance.json').write_text(json.dumps(core.serialize(ins),indent=2))
    print(json.dumps({'case':info,'methods':rows},indent=2),flush=True)
    return rows,info


def run_suite(out,repetitions=5,training_trips=3000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=core.synthetic_instance(); rows=[];diagnostics=[]; case_id=0
    npz_inputs={};start=time.perf_counter()
    for constraints,gap in [('base',.23),('tight',.18)]:
        ins0=replace(base,max_gap=gap)
        layouts,total=enumerate_feasible(ins0)
        incidence,n1,n2,components,labels=incidence_and_neighbors(layouts,len(ins0.candidates))
        core.save_csv(out/f'feasible_{constraints}.csv',[{'index':i,'stops':';'.join(map(str,s))} for i,s in enumerate(layouts)])
        uniform_indices,_=projection_indices(ins0,layouts,np.full(128,ins0.budget),np.linspace(0,1,129),np.arange(16)/16)
        for dwell_type in ('constant','heterogeneous'):
            mu=np.full_like(base.dwell,base.dwell.mean()) if dwell_type=='constant' else base.dwell.copy()
            ins=replace(ins0,dwell=mu)
            for service,dirs in SERVICES.items():
                costs=np.array([journey_costs(ins,s,dirs)[0].ravel() for s in layouts])
                for family in ('uniform','local','hubs','commuter'):
                    truth=population(ins.zones,family);popcost=costs@truth.ravel()
                    ref=int(np.flatnonzero(np.all(layouts==ins.reference,axis=1))[0])
                    for replicate in range(repetitions):
                        case_id+=1;seed=730000+case_id
                        rng=np.random.default_rng(seed)
                        traincounts=rng.multinomial(training_trips,truth.ravel()).reshape(truth.shape)
                        testcounts=rng.multinomial(12000,truth.ravel()).reshape(truth.shape)
                        npz_inputs[f'train_{case_id}']=traincounts;npz_inputs[f'test_{case_id}']=testcounts
                        train=costs@core.probabilities(traincounts).ravel();test=costs@core.probabilities(testcounts).ravel()
                        methods,info,G,_=select_methods(ins,layouts,train,traincounts,dirs,n1,n2,uniform_indices)
                        meta={'case':case_id,'seed':seed,'constraints':constraints,'dwell':dwell_type,'service':service,'demand':family,'replicate':replicate}
                        for name,j in methods.items():
                            rows.append({**meta,'method':name,'stops':';'.join(map(str,layouts[j])),
                                'population_minutes':float(popcost[j]),'population_regret_seconds':60*float(popcost[j]-popcost.min()),
                                'population_saving_seconds':60*float(popcost[ref]-popcost[j]),
                                'test_saving_seconds':60*float(test[ref]-test[j]),
                                'train_regret_seconds':60*float(train[j]-train.min())})
                        diagnostics.append({**meta,'feasible_layouts':len(layouts),'one_swap_components':int(components),**info})
                    print(f'Completed {constraints}/{dwell_type}/{service}/{family}: {case_id} fits, {time.perf_counter()-start:.1f}s',flush=True)
                    core.save_csv(out/'suite_results.csv',rows);core.save_csv(out/'suite_diagnostics.csv',diagnostics)
    np.savez_compressed(out/'suite_counts.npz',**npz_inputs)
    summary=[]
    for name in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==name]
        regrets=np.array([r['population_regret_seconds'] for r in rr]);sav=np.array([r['population_saving_seconds'] for r in rr]);tr=np.array([r['train_regret_seconds'] for r in rr])
        summary.append({'method':name,'fits':len(rr),'mean_population_regret_seconds':float(regrets.mean()),
                        'median_population_regret_seconds':float(np.median(regrets)),
                        'p95_population_regret_seconds':float(np.quantile(regrets,.95)),
                        'max_population_regret_seconds':float(regrets.max()),
                        'mean_population_saving_seconds':float(sav.mean()),
                        'worse_than_reference_count':int((sav < -1e-8).sum()),
                        'training_exact_count':int((tr<1e-6).sum())})
    core.save_csv(out/'suite_summary.csv',summary)
    meta={'structural_scenarios':48,'repetitions':repetitions,'fits':case_id,'training_trips':training_trips,'test_trips':12000,
          'candidate_count':24,'zones':32,'budget':8,'phases':16,'elapsed_seconds':time.perf_counter()-start,
          'population_reference':'Known generating distribution evaluated exactly on the 32-zone finite OD support. Never used for design selection.',
          'interpretation':'Synthetic factorial stress test, not a random sample of real transit networks. Population oracle and training optimum are different comparators.'}
    (out/'suite_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


class ExperimentTests(unittest.TestCase):
    def test_directional_matches_old(self):
        ins=core.synthetic_instance()
        for stops in [ins.reference,(0,2,4,8,12,14,16,20)]:
            a,ab=journey_costs(ins,stops);b,bb=core.trip_costs(ins,stops)
            np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_array_equal(ab,bb)
        a,B,c,constant,*_=frozen_coefficients(ins,ins.train_counts,SERVICES['symmetric'])
        aa,BB,cc=core.design_coefficients(ins,ins.train_counts)
        np.testing.assert_allclose(a,aa);np.testing.assert_allclose(B,BB);np.testing.assert_allclose(c,cc)
    def test_one_way_hand_case(self):
        z=np.array([0.,.25,.5,.75]);counts=np.ones((4,4));np.fill_diagonal(counts,0)
        ins=core.Instance(z,z,np.full(4,1.),counts,counts,(0,1,2,3),budget=4,mandatory=(),forbidden=(),min_gap=0,max_gap=.4,
                         max_walk_minutes=100,walking_minutes_per_loop=1000,motion_minutes_per_loop=12)
        c,_=journey_costs(ins,ins.reference,(Direction(1,2,12),))
        self.assertAlmostEqual(c[0,1],8)
        self.assertAlmostEqual(c[1,0],16)
    def test_unequal_fleet_hand_case(self):
        z=np.array([0.,.25,.5,.75]);counts=np.ones((4,4));np.fill_diagonal(counts,0)
        ins=core.Instance(z,z,np.full(4,1.),counts,counts,(0,1,2,3),budget=4,mandatory=(),forbidden=(),
                         min_gap=0,max_gap=.4,max_walk_minutes=100,walking_minutes_per_loop=1000)
        c,_=journey_costs(ins,ins.reference,(Direction(1,4,12),Direction(-1,1,12)))
        self.assertAlmostEqual(c[0,1],6);self.assertAlmostEqual(c[1,0],12)
    def test_fast_vs_independent_paths(self):
        ins=core.synthetic_instance();s=list(ins.reference);directions=SERVICES['unequal']
        fast,_=journey_costs(ins,s,directions)
        for r,t in [(2,11),(13,4),(27,19)]:
            best=90*core.circular_distance(ins.zones[r],ins.zones[t])
            for a in s:
                for b in s:
                    if a==b:continue
                    for d in directions:
                        length=(d.sign*(ins.candidates[b]-ins.candidates[a]))%1
                        interior=[j for j in s if EPS<(d.sign*(ins.candidates[j]-ins.candidates[a]))%1<length-EPS]
                        ride=d.motion*length+d.dwell_scale*(sum(ins.dwell[j] for j in interior)+.5*(ins.dwell[a]+ins.dwell[b]))
                        wait=(d.motion+d.dwell_scale*sum(ins.dwell[j] for j in s))/(2*d.fleet)
                        walk=90*(core.circular_distance(ins.zones[r],ins.candidates[a])+core.circular_distance(ins.zones[t],ins.candidates[b]))
                        best=min(best,ride+wait+walk)
            self.assertAlmostEqual(fast[r,t],best,places=10)
    def test_bounded_density(self):
        A=np.array([1.,20.,3.]);B=np.array([2.,.1,5.]);w=np.array([.3,.2,.5]);lower=np.array([1.,1.,1.]);upper=np.array([3.,4.,5.])
        n,lam=bounded_density(A,B,w,2.8,lower,upper)
        self.assertAlmostEqual(w@n,2.8,places=9)
        self.assertTrue(np.all(n>=lower));self.assertTrue(np.all(n<=upper))
        from scipy.optimize import minimize
        sol=minimize(lambda x:w@(A/x+B*x),np.full(3,2.8),bounds=list(zip(lower,upper)),
                     constraints={'type':'eq','fun':lambda x:w@x-2.8},method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
        self.assertTrue(sol.success);self.assertAlmostEqual(w@(A/n+B*n),sol.fun,places=8)
        with self.assertRaises(ValueError):bounded_density(A,B,w,6,lower,upper)
    def test_projection_matches_milp(self):
        ins=core.synthetic_instance();layouts,_=enumerate_feasible(ins)
        fixed,_,edges,_=density_profiles(ins,ins.train_counts,SERVICES['symmetric'])
        indices,values=projection_indices(ins,layouts,fixed,edges,[.125,.5,.875])
        for phase,idx,val in zip([.125,.5,.875],indices,values):
            cdf=np.r_[0,np.cumsum(fixed*np.diff(edges))]/ins.budget
            targets=np.interp((np.arange(ins.budget)+phase)/ins.budget,cdf,edges)
            cost=core.circular_distance(targets[:,None],ins.candidates[None,:])
            stops,obj,_=core.solve_design(ins,np.ones(ins.budget),cost,np.zeros(len(ins.candidates)),True)
            self.assertAlmostEqual(obj,val,places=8)
    def test_population_and_count(self):
        ins=core.synthetic_instance();layouts,_=enumerate_feasible(ins)
        self.assertEqual(len(layouts),508)
        for family in ('uniform','local','hubs','commuter'):
            p=population(ins.zones,family);self.assertAlmostEqual(p.sum(),1);self.assertTrue(np.all(np.diag(p)==0))
    def test_transfer_bound(self):
        rng=np.random.default_rng(55)
        for _ in range(30):
            G=rng.normal(size=10);F=rng.normal(size=10);e=F-G
            self.assertTrue(np.all(F-F.min()<=G-G.min()+np.ptp(e)+EPS))
    def test_order_preserving_snapping_bound(self):
        rng=np.random.default_rng(123);zones=(np.arange(20)+.5)/20
        u=(np.arange(8)+.5)/8; v=u+rng.uniform(-.006,.006,8)
        mu=lambda a:.25+.1*np.sin(2*np.pi*a)
        od=np.ones((20,20));np.fill_diagonal(od,0)
        ins=core.Instance(u,zones,mu(u),od,od,tuple(range(8)),budget=8,mandatory=(),forbidden=(),
            min_gap=0,max_gap=.4,max_walk_minutes=100)
        other=replace(ins,candidates=v,dwell=mu(v))
        delta=float(np.max(np.abs(v-u)));lip=.2*np.pi
        for directions in SERVICES.values():
            a,_=journey_costs(ins,ins.reference,directions)
            b,_=journey_costs(other,other.reference,directions)
            bound=delta*max(2*90+2*d.motion+(7+8/(2*d.fleet))*lip*d.dwell_scale for d in directions)
            self.assertLessEqual(float(np.max(np.abs(a-b))),bound+1e-10)

    def test_generalized_surrogate_milp(self):
        ins=core.synthetic_instance();layouts,_=enumerate_feasible(ins)
        for directions in SERVICES.values():
            a,B,c,constant,*_=frozen_coefficients(ins,ins.train_counts,directions)
            chosen,value,gap=core.solve_design(ins,a,c,B)
            brute=min(float(a@c[:,ss].min(axis=1)+B[ss].sum()) for ss in layouts)
            self.assertAlmostEqual(value,brute,places=8)
            self.assertLessEqual(gap,1e-8)

    def test_one_way_wait_coefficient(self):
        ins=core.synthetic_instance();od=ins.train_counts
        *_,flows,shares=frozen_coefficients(ins,od,SERVICES['clockwise'])
        self.assertAlmostEqual(shares[0],1.)
        _,B,_,_,q,_=frozen_coefficients(ins,od,SERVICES['clockwise'])
        np.testing.assert_allclose(B,ins.dwell*(1/6+q[0]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='store_true');parser.add_argument('--suite',action='store_true');parser.add_argument('--test',action='store_true')
    parser.add_argument('--out',default='results');parser.add_argument('--repetitions',type=int,default=5);parser.add_argument('--training-trips',type=int,default=3000)
    args=parser.parse_args()
    if args.test:
        unittest.main(argv=[sys.argv[0]])
    else:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},indent=2))
        if args.case:run_case(out)
        if args.suite:run_suite(out,args.repetitions,args.training_trips)
