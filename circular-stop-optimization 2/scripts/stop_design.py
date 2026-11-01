#!/usr/bin/env python3
"""Circular stop design and finite-candidate optimization."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, milp
from scipy.sparse import coo_matrix

EPS = 1e-9


def circular_distance(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(x, 1.0 - x)


def probabilities(counts):
    z = np.asarray(counts, dtype=float)
    if not np.all(np.isfinite(z)) or np.any(z < 0) or z.sum() <= 0:
        raise ValueError('Demand must be finite, nonnegative, and have positive total.')
    return z / z.sum()


def fixed_budget_density(A, B, widths, budget):
    """Solve the finite-grid fixed-budget problem."""
    A, B, widths = [np.asarray(v, dtype=float) for v in (A, B, widths)]
    if A.ndim != 1 or A.shape != B.shape or A.shape != widths.shape:
        raise ValueError('A, B, widths must be same-length one-dimensional arrays.')
    if not all(np.all(np.isfinite(v)) for v in (A, B, widths)):
        raise ValueError('Inputs must be finite.')
    if np.any(A <= 0) or np.any(B < 0) or np.any(widths <= 0) or budget <= 0:
        raise ValueError('Require A>0, B>=0, widths>0 and budget>0.')
    shifted = B - B.min()
    def mass(t):
        return float(widths @ np.sqrt(A / (shifted + t)))
    lower = 1e-15
    while mass(lower) < budget:
        lower *= 1e-2
        if lower < 1e-290:
            raise ValueError('Numerically extreme stop budget.')
    upper = max(1.0, (float(widths @ np.sqrt(A)) / budget) ** 2)
    while mass(upper) > budget:
        upper *= 2
    t = brentq(lambda x: mass(x) - budget, lower, upper,
               xtol=1e-14, rtol=1e-13)
    n = np.sqrt(A / (shifted + t))
    if abs(float(widths @ n) - budget) > 1e-7 * max(1, budget):
        raise RuntimeError('Fixed-budget root did not reach the required tolerance.')
    return n, float(t - B.min())


def continuum_cost(A, B, widths, n):
    return float(np.asarray(widths) @ (np.asarray(A) / n + np.asarray(B) * n))


@dataclass
class Instance:
    candidates: np.ndarray
    zones: np.ndarray
    dwell: np.ndarray
    train_counts: np.ndarray
    test_counts: np.ndarray
    reference: tuple[int, ...]
    budget: int = 8
    mandatory: tuple[int, ...] = (0, 12)
    forbidden: tuple[int, ...] = (5, 17)
    min_gap: float = .055
    max_gap: float = .23
    max_walk_minutes: float = 12.0
    walking_minutes_per_loop: float = 90.0
    motion_minutes_per_loop: float = 18.0
    fleet_total: int = 6
    provenance: str = 'UNSPECIFIED'

    def validate(self):
        m, p = len(self.candidates), len(self.zones)
        for name, v in [('candidates', self.candidates), ('zones', self.zones)]:
            if v.ndim != 1 or not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v >= 1):
                raise ValueError(f'{name} must be a finite 1-D array in [0,1).')
            if np.any(np.diff(v) <= 0):
                raise ValueError(f'{name} must be strictly increasing.')
        if self.dwell.shape != (m,) or not np.all(np.isfinite(self.dwell)) or np.any(self.dwell <= 0):
            raise ValueError('One positive finite dwell per candidate is required.')
        for counts in [self.train_counts, self.test_counts]:
            if counts.shape != (p, p):
                raise ValueError('OD counts must have shape (number of zones, number of zones).')
            probabilities(counts)
            if not np.allclose(counts, np.round(counts), atol=1e-8):
                raise ValueError('Bootstrap inputs must be integer trip counts, not uncalibrated OD weights.')
        if not 2 <= self.budget <= m or self.fleet_total < 2 or self.fleet_total % 2:
            raise ValueError('Require 2<=budget<=candidates and a positive even total fleet.')
        if not 0 <= self.min_gap <= self.max_gap <= 1:
            raise ValueError('Gap limits must satisfy 0<=minimum<=maximum<=1.')
        if min(self.walking_minutes_per_loop, self.motion_minutes_per_loop, self.max_walk_minutes) <= 0:
            raise ValueError('Walking, motion and accessibility limits must be positive.')
        if set(self.mandatory) & set(self.forbidden):
            raise ValueError('A candidate cannot be mandatory and forbidden.')
        for j in (*self.mandatory, *self.forbidden, *self.reference):
            if not isinstance(j, (int, np.integer)) or not 0 <= j < m:
                raise ValueError('Stop indices are out of range.')
        if not feasible(self, self.reference):
            raise ValueError('Reference layout violates the specified constraints.')


def feasible(ins, stops):
    s = tuple(sorted(stops))
    if len(s) != ins.budget or len(set(s)) != len(s):
        return False
    if not set(ins.mandatory).issubset(s) or set(ins.forbidden) & set(s):
        return False
    u = ins.candidates[list(s)]
    gaps = np.diff(np.r_[u, u[0]+1])
    if gaps.min() < ins.min_gap-EPS or gaps.max() > ins.max_gap+EPS:
        return False
    d = circular_distance(ins.zones[:, None], u[None, :])
    return bool(np.max(d.min(axis=1))*ins.walking_minutes_per_loop <= ins.max_walk_minutes+EPS)


def flow_at(points, zones, od):
    """Compute directional through-flow on shortest arcs."""
    p = probabilities(od)
    x = zones[:, None]
    y = zones[None, :]
    length = (y-x) % 1
    cw_share = np.where(length < .5-EPS, 1., np.where(length > .5+EPS, 0., .5))
    cw_share[np.eye(len(zones), dtype=bool)] = 0
    ccw_share = 1-cw_share
    ccw_share[np.eye(len(zones), dtype=bool)] = 0
    cw, ccw = [], []
    for u in points:
        a = (u-x) % 1
        b = (x-u) % 1
        cw.append(np.sum(p*cw_share*(a > EPS)*(a < length-EPS)))
        ccw.append(np.sum(p*ccw_share*(b > EPS)*(b < (1-length)-EPS)))
    return np.array(cw), np.array(ccw)


def design_coefficients(ins, od):
    p = probabilities(od)
    endpoint_mass = p.sum(axis=0)+p.sum(axis=1)
    q1, q2 = flow_at(ins.candidates, ins.zones, p)
    stop_penalty = ins.dwell*(1/ins.fleet_total+q1+q2)
    walk = ins.walking_minutes_per_loop*circular_distance(ins.zones[:, None], ins.candidates[None, :])
    return endpoint_mass, stop_penalty, walk


def solve_design(ins, weights, access_cost, stop_penalty, one_to_one=False):
    """Solve the finite frozen-flow mixed-integer program."""
    weights = np.asarray(weights, float)
    access_cost = np.asarray(access_cost, float)
    stop_penalty = np.asarray(stop_penalty, float)
    r, m = access_cost.shape
    if m != len(ins.candidates) or weights.shape != (r,) or stop_penalty.shape != (m,):
        raise ValueError('Coefficient dimensions do not match the candidate set.')
    if np.any(weights < 0) or not np.all(np.isfinite(access_cost)):
        raise ValueError('Invalid cost inputs.')
    if one_to_one and r != ins.budget:
        raise ValueError('Quantile projection requires exactly budget targets.')
    nvar = m+r*m
    objective = np.r_[stop_penalty, (weights[:,None]*access_cost).ravel()]
    rows, cols, vals, lower, upper = [], [], [], [], []
    def add(c, v, lo=-np.inf, hi=np.inf):
        idx = len(lower)
        rows.extend([idx]*len(c)); cols.extend(c); vals.extend(v)
        lower.append(lo); upper.append(hi)
    add(list(range(m)), [1.]*m, ins.budget, ins.budget)
    for i in range(r):
        add(list(range(m+i*m, m+(i+1)*m)), [1.]*m, 1, 1)
        for j in range(m):
            add([m+i*m+j, j], [1., -1.], hi=0)
    if one_to_one:
        for j in range(m):
            add([m+i*m+j for i in range(r)]+[j], [1.]*r+[-1.], 0, 0)
    for j in range(m):
        for l in range(j+1,m):
            if circular_distance(ins.candidates[j],ins.candidates[l]) < ins.min_gap-EPS:
                add([j,l], [1.,1.], hi=1)
        forward = (ins.candidates-ins.candidates[j]) % 1
        window = np.flatnonzero((forward > EPS)&(forward <= ins.max_gap+EPS)).tolist()
        add(window, [1.]*len(window), lo=1)
    real_access = ins.walking_minutes_per_loop*circular_distance(ins.zones[:,None],ins.candidates[None,:])
    for i in range(len(ins.zones)):
        allowed = np.flatnonzero(real_access[i] <= ins.max_walk_minutes+EPS).tolist()
        add(allowed, [1.]*len(allowed), lo=1)
    lb, ub = np.zeros(nvar), np.ones(nvar)
    lb[list(ins.mandatory)] = 1
    ub[list(ins.forbidden)] = 0
    integrality = np.zeros(nvar); integrality[:m] = 1
    mat = coo_matrix((vals,(rows,cols)), shape=(len(lower),nvar)).tocsc()
    result = milp(objective, integrality=integrality, bounds=Bounds(lb,ub),
                  constraints=LinearConstraint(mat,np.array(lower),np.array(upper)),
                  options={'mip_rel_gap':1e-9,'time_limit':90.0})
    if result.status != 0 or result.x is None:
        raise RuntimeError(f'MILP did not certify optimality: {result.message}')
    chosen = tuple(np.flatnonzero(result.x[:m]>.5).tolist())
    if not feasible(ins,chosen):
        raise RuntimeError('MILP output failed an independent feasibility check.')
    return chosen, float(result.fun), float(result.mip_gap)


def surrogate_cost(ins, stops, od, include_constant=True):
    a, B, c = design_coefficients(ins,od)
    val = float(a @ c[:,list(stops)].min(axis=1)+B[list(stops)].sum())
    if include_constant:
        distances = circular_distance(ins.zones[:,None],ins.zones[None,:])
        val += ins.motion_minutes_per_loop*(1/ins.fleet_total+np.sum(probabilities(od)*distances))
    return val


def trip_costs(ins, stops, walk_scale=1., dwell_scale=1.):
    """Evaluate expected door-to-door travel time for each OD pair."""
    s = list(sorted(stops))
    u, mu = ins.candidates[s], ins.dwell[s]*dwell_scale
    n = len(s)
    ride = np.full((n,n),np.inf)
    cycle = ins.motion_minutes_per_loop+mu.sum()
    for j in range(n):
        for l in range(n):
            if j == l:
                continue
            arc = (u[l]-u[j]) % 1
            forward = (u-u[j]) % 1
            interior = (forward > EPS)&(forward < arc-EPS)
            cw = ins.motion_minutes_per_loop*arc + mu[interior].sum()+.5*(mu[j]+mu[l])
            ccw = cycle-cw
            ride[j,l] = min(cw,ccw)+cycle/ins.fleet_total
    walking = ins.walking_minutes_per_loop*walk_scale
    access = walking*circular_distance(ins.zones[:,None],u[None,:])
    bus = (access[:,None,:,None]+access[None,:,None,:]+ride[None,None,:,:]).min(axis=(2,3))
    direct = walking*circular_distance(ins.zones[:,None],ins.zones[None,:])
    return np.minimum(bus,direct), bus < direct-EPS


def swap_search(ins, initial, od):
    """Find a local optimum using one-stop exchanges."""
    p = probabilities(od)
    current = tuple(sorted(initial))
    value = float(np.sum(p*trip_costs(ins,current)[0]))
    steps = 0
    while True:
        best, bestvalue = current, value
        for old in current:
            if old in ins.mandatory:
                continue
            for new in range(len(ins.candidates)):
                if new in current or new in ins.forbidden:
                    continue
                trial = tuple(sorted((set(current)-{old})|{new}))
                if not feasible(ins,trial):
                    continue
                v = float(np.sum(p*trip_costs(ins,trial)[0]))
                if v < bestvalue-1e-10:
                    best,bestvalue = trial,v
        if best == current:
            return current,value,steps
        current,value = best,bestvalue
        steps += 1
        if steps > 1000:
            raise RuntimeError('Unexpectedly long local search.')


def exhaustive_reassignment(ins, od, limit=100000):
    """Enumerate feasible layouts in small candidate sets."""
    import math
    optional = [j for j in range(len(ins.candidates))
                if j not in ins.mandatory and j not in ins.forbidden]
    remaining = ins.budget-len(ins.mandatory)
    if remaining < 0 or remaining > len(optional):
        raise ValueError('Infeasible mandatory/forbidden stop budget.')
    total = math.comb(len(optional),remaining)
    if total > limit:
        return None, {'status':'skipped','candidate_subsets':total,'limit':limit}
    p = probabilities(od)
    best, value, checked = None, np.inf, 0
    for extra in itertools.combinations(optional,remaining):
        stops = tuple(sorted((*ins.mandatory,*extra)))
        if not feasible(ins,stops):
            continue
        checked += 1
        trial = float(np.sum(p*trip_costs(ins,stops)[0]))
        if trial < value-1e-10:
            best,value = stops,trial
    if best is None:
        raise ValueError('No feasible layout in exhaustive enumeration.')
    return best, {'status':'complete','candidate_subsets':total,
                  'feasible_layouts':checked,'train_optimal_minutes':value}


def density_designs(ins, od, bins=128):
    p = probabilities(od)
    endpoints = p.sum(axis=0)+p.sum(axis=1)
    masses, edges = np.histogram(ins.zones,bins=bins,range=(0,1),weights=endpoints)
    from scipy.ndimage import gaussian_filter1d
    masses = gaussian_filter1d(masses,2.,mode='wrap')
    masses = .99*masses+.01*(2/bins)
    u = (edges[1:]+edges[:-1])/2
    mu = np.interp(u, np.r_[ins.candidates[-1]-1,ins.candidates,ins.candidates[0]+1],
                      np.r_[ins.dwell[-1],ins.dwell,ins.dwell[0]])
    q1,q2 = flow_at(u,ins.zones,p)
    A = ins.walking_minutes_per_loop*(masses*bins)/4
    B = mu*(1/ins.fleet_total+q1+q2)
    widths = np.full(bins,1/bins)
    free = np.sqrt(A/B)
    normalized = free*ins.budget/(widths@free)
    fixed,lam = fixed_budget_density(A,B,widths,ins.budget)
    layouts = {}
    for label,density in [('rescaled_free',normalized),('fixed_budget',fixed)]:
        cdf = np.r_[0,np.cumsum(density*widths)]/ins.budget
        targets = np.interp((np.arange(ins.budget)+.5)/ins.budget,cdf,edges)
        costs = circular_distance(targets[:,None],ins.candidates[None,:])
        layouts[label] = solve_design(ins,np.ones(ins.budget),costs,np.zeros(len(ins.candidates)),True)[0]
    gap = continuum_cost(A,B,widths,normalized)-continuum_cost(A,B,widths,fixed)
    identity = float(widths@((B+lam)*(normalized-fixed)**2/normalized))
    info = {'multiplier':lam, 'free_stop_mass':float(widths@free),
            'fixed_budget_mass':float(widths@fixed),
            'rescaled_continuum_cost':continuum_cost(A,B,widths,normalized),
            'optimal_continuum_cost':continuum_cost(A,B,widths,fixed),
            'continuum_cost_gap':gap, 'gap_identity_rhs':identity,
            'gap_identity_error':abs(gap-identity),
            'endpoint_smoothing':'128 bins; circular Gaussian sigma=2 bins; 1% uniform component'}
    return layouts,info


def synthetic_instance(seed=20261031):
    rng = np.random.default_rng(seed)
    m,p = 24,32
    candidates = np.arange(m)/m
    zones = (np.arange(p)+.5)/p
    def bump(x,c,s):
        return np.exp(-.5*(circular_distance(x,c)/s)**2)
    origins = .15+2.5*bump(zones,.13,.065)+1.8*bump(zones,.59,.075)
    destinations = .15+2.2*bump(zones,.34,.065)+1.7*bump(zones,.85,.065)
    joint = np.outer(origins,destinations)
    joint = .7*probabilities(joint)+.3*probabilities(np.exp(-circular_distance(zones[:,None],zones[None,:])/.12))
    np.fill_diagonal(joint,0)
    joint = probabilities(joint)
    train = rng.multinomial(3000,joint.ravel()).reshape(p,p)
    test = rng.multinomial(12000,joint.ravel()).reshape(p,p)
    dwell = .18+.55*bump(candidates,.39,.08)+.23*bump(candidates,.78,.10)
    ins = Instance(candidates,zones,dwell,train,test,tuple(range(0,m,3)),
                   provenance='SYNTHETIC two-direction 6 km loop; seed 20261031. Not UCSD data.')
    ins.validate()
    return ins


def serialize(ins):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(ins).items()}


def load_instance(path):
    data = json.loads(Path(path).read_text())
    for k in ['candidates','zones','dwell','train_counts','test_counts']:
        data[k] = np.array(data[k],float)
    for k in ['reference','mandatory','forbidden']:
        data[k] = tuple(data[k])
    ins = Instance(**data); ins.validate(); return ins


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(ins, out, bootstrap_repetitions=500):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    ins.validate()
    ptrain,ptest = probabilities(ins.train_counts),probabilities(ins.test_counts)
    a,B,c = design_coefficients(ins,ptrain)
    exact,_,mipgap = solve_design(ins,a,c,B)
    access,_,_ = solve_design(ins,a,c,np.zeros_like(B))
    layouts,continuum = density_designs(ins,ptrain)
    layouts = {'reference':ins.reference,**layouts,'access_only':access,'frozen_flow_milp':exact}
    searched = [swap_search(ins,s,ptrain) for s in dict.fromkeys(layouts.values())]
    best = min(searched,key=lambda t:t[1])
    layouts['reassignment_local_search'] = best[0]
    enumerated,enumeration_info=exhaustive_reassignment(ins,ptrain)
    if enumerated is not None:
        layouts['reassignment_exact_enumeration']=enumerated
    reference_cost = trip_costs(ins,ins.reference)[0]
    rng=np.random.default_rng(121)
    ntest=int(round(ins.test_counts.sum()))
    draws=rng.multinomial(ntest,ptest.ravel(),size=bootstrap_repetitions)/ntest
    rows=[]
    matrices={}
    for name,s in layouts.items():
        costs,bus=trip_costs(ins,s); matrices[name]=costs
        improvement=reference_cost-costs
        intervals=np.quantile(draws@improvement.ravel(),[.025,.975])
        rows.append({'method':name,'stops':';'.join(map(str,s)),
          'train_surrogate_minutes':surrogate_cost(ins,s,ptrain),
          'train_reassigned_minutes':float(np.sum(ptrain*costs)),
          'test_reassigned_minutes':float(np.sum(ptest*costs)),
          'test_saving_seconds':60*float(np.sum(ptest*improvement)),
          'saving_ci_low_seconds':60*float(intervals[0]),
          'saving_ci_high_seconds':60*float(intervals[1]),
          'test_bus_share':float(np.sum(ptest*bus)),
          'max_nearest_walk_minutes':float(ins.walking_minutes_per_loop*circular_distance(ins.zones[:,None],ins.candidates[list(s)][None,:]).min(axis=1).max())})
    save_csv(out/'benchmark.csv',rows)
    sensitivity=[]
    for w in [.8,1.,1.2]:
        for d in [.5,1.,1.5]:
            ref=trip_costs(ins,ins.reference,w,d)[0]
            for name,s in layouts.items():
                costs=trip_costs(ins,s,w,d)[0]
                sensitivity.append({'walking_scale':w,'dwell_scale':d,'method':name,
                    'test_mean_minutes':float(np.sum(ptest*costs)),
                    'saving_seconds':60*float(np.sum(ptest*(ref-costs)))})
    save_csv(out/'sensitivity.csv',sensitivity)
    stability=np.zeros(len(ins.candidates)); bootstrap_layouts=[]
    for b in range(30):
        counts=rng.multinomial(int(ins.train_counts.sum()),ptrain.ravel()).reshape(ptrain.shape)
        aa,bb,cc=design_coefficients(ins,counts)
        ss,_,_=solve_design(ins,aa,cc,bb)
        stability[list(ss)]+=1
        bootstrap_layouts.append({'replicate':b,'stops':';'.join(map(str,ss))})
    save_csv(out/'selection_stability.csv',[{'candidate':j,'coordinate':float(ins.candidates[j]),
             'fraction_selected':float(v/30)} for j,v in enumerate(stability)])
    save_csv(out/'bootstrap_layouts.csv',bootstrap_layouts)
    metadata={'provenance':ins.provenance,'mip_relative_gap':mipgap,
              'continuum':continuum,'bootstrap_repetitions':bootstrap_repetitions,
              'training_bootstrap_refits':30,'n_train':float(ins.train_counts.sum()),
              'n_test':float(ins.test_counts.sum()),'local_search_starts':len(searched),'exhaustive_reassignment':enumeration_info,
              'interpretation':'All demo results synthetic. Test uncertainty is conditional on fixed designs and model; it does not include model misspecification, arrival noise or real-world sampling.'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'instance.json').write_text(json.dumps(serialize(ins),indent=2))
    print(json.dumps({'metadata':metadata,'benchmark':rows},indent=2))
    return rows,metadata


class Tests(unittest.TestCase):
    def test_budget_and_exact_gap(self):
        A=np.array([1.,3.,9.]); B=np.array([.3,2.,7.]); widths=np.array([.2,.3,.5])
        n,lam=fixed_budget_density(A,B,widths,4)
        self.assertAlmostEqual(float(widths@n),4,places=9)
        candidate=np.full(3,4.)
        lhs=continuum_cost(A,B,widths,candidate)-continuum_cost(A,B,widths,n)
        rhs=float(widths@((B+lam)*(candidate-n)**2/candidate))
        self.assertAlmostEqual(lhs,rhs,places=9)
        self.assertGreaterEqual(lhs,-1e-10)

    def test_constant_B(self):
        A=np.array([1.,4.,9.]); B=np.full(3,2.); widths=np.full(3,1/3)
        n,_=fixed_budget_density(A,B,widths,6)
        np.testing.assert_allclose(n,6*np.sqrt(A)/(widths@np.sqrt(A)))

    def test_milp_against_exhaustive(self):
        for seed in range(4):
            rng=np.random.default_rng(seed)
            m,p=10,7
            od=rng.integers(1,20,size=(p,p)).astype(float);np.fill_diagonal(od,0)
            ins=Instance(np.arange(m)/m,(np.arange(p)+.5)/p,.1+rng.random(m),od,od,
                (0,3,6),budget=3,mandatory=(0,),forbidden=(8,),min_gap=.1,max_gap=.5,
                max_walk_minutes=30.)
            ins.validate();a,B,c=design_coefficients(ins,od)
            chosen,value,_=solve_design(ins,a,c,B)
            brute=min(surrogate_cost(ins,s,od,False)
                for s in itertools.combinations(range(m),3) if feasible(ins,s))
            self.assertAlmostEqual(value,brute,places=7)
            self.assertAlmostEqual(surrogate_cost(ins,chosen,od,False),brute,places=7)

    def test_trip_evaluator(self):
        ins=synthetic_instance();costs,bus=trip_costs(ins,ins.reference)
        np.testing.assert_allclose(costs,costs.T,atol=1e-10)
        np.testing.assert_allclose(np.diag(costs),0)
        direct=ins.walking_minutes_per_loop*circular_distance(ins.zones[:,None],ins.zones[None,:])
        self.assertTrue(np.all(costs<=direct+EPS))
        expensive=trip_costs(ins,ins.reference,dwell_scale=2.)[0]
        self.assertTrue(np.all(expensive>=costs-EPS))

    def test_flow_integral(self):
        zones=np.array([.1,.4,.7]);od=np.array([[0.,4.,1.],[2.,0.,3.],[1.,2.,0.]])
        points=(np.arange(12000)+.5)/12000
        q1,q2=flow_at(points,zones,od)
        expected=np.sum(probabilities(od)*circular_distance(zones[:,None],zones[None,:]))
        self.assertAlmostEqual(float((q1+q2).mean()),float(expected),places=4)

    def test_multiplier_sign(self):
        A=np.array([1.,4.]); B=np.array([1.,3.]); widths=np.array([.5,.5])
        free=float(widths@np.sqrt(A/B))
        self.assertGreater(fixed_budget_density(A,B,widths,.5*free)[1],0)
        self.assertLess(fixed_budget_density(A,B,widths,2*free)[1],0)

    def test_zero_access_guard(self):
        with self.assertRaises(ValueError):
            fixed_budget_density([1.,0.],[1.,2.],[.5,.5],4)

    def test_quantile_projection(self):
        ins=synthetic_instance()
        layouts, info=density_designs(ins,ins.train_counts)
        for stops in layouts.values():
            self.assertTrue(feasible(ins,stops))
            self.assertEqual(len(set(stops)),ins.budget)
        self.assertLess(info['gap_identity_error'],1e-8)

    def test_exhaustive_and_local_search(self):
        rng=np.random.default_rng(4)
        m,p=10,7
        od=rng.integers(1,20,size=(p,p)).astype(float);np.fill_diagonal(od,0)
        ins=Instance(np.arange(m)/m,(np.arange(p)+.5)/p,.1+rng.random(m),od,od,
            (0,3,6),budget=3,mandatory=(0,),forbidden=(8,),min_gap=.1,max_gap=.5,
            max_walk_minutes=30.)
        global_stops,info=exhaustive_reassignment(ins,od)
        local_stops,value,_=swap_search(ins,ins.reference,od)
        self.assertTrue(feasible(ins,global_stops))
        self.assertLessEqual(info['train_optimal_minutes'],value+1e-9)

    def test_infeasible_design_rejected(self):
        ins=synthetic_instance()
        broken=replace(ins,budget=3,min_gap=.4,max_gap=.45)
        a,B,c=design_coefficients(broken,broken.train_counts)
        with self.assertRaises(RuntimeError):
            solve_design(broken,a,c,B)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--test',action='store_true')
    args=parser.parse_args()
    if args.test:
        unittest.main(argv=['stop_design.py'],exit=True)
    elif args.demo or args.input:
        run(load_instance(args.input) if args.input else synthetic_instance(),args.out)
    else:
        parser.print_help()
