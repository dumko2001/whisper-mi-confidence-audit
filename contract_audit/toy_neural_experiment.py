#!/usr/bin/env python3
"""Original local mechanism experiment, NOT upstream estimator reproduction.

Uses standard correctly normalized DV gradients rather than the upstream extra
batch-size division. No external repository module is imported or executed.
The precommitted experiment_protocol.json controls this one-shot experiment.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = "a975fe7165f79e423c28982bc1bd6c7ce0f533c8199661bd225a1532dd3ed455"


def logmeanexp(scores):
    high = float(np.max(scores))
    return high + math.log(float(np.mean(np.exp(scores-high))))


def forward(parameters, rows):
    pre = rows @ parameters["a"] + parameters["b"]
    hidden = np.maximum(pre, 0.)
    return (hidden @ parameters["c"] + parameters["d"]).ravel(), pre, hidden


def objective_and_gradient(parameters, joint, reference):
    """Return NEGATIVE DV and standard exact gradients, one mean division only."""
    rows = np.concatenate((joint,reference))
    scores, pre, hidden = forward(parameters,rows)
    nj = len(joint)
    q = scores[nj:]
    shifted = np.exp(q - np.max(q))
    softmax = shifted / np.sum(shifted)
    derivative = np.concatenate((np.full(nj,-1./nj),softmax))[:,None]
    hidden_derivative = (derivative @ parameters["c"].T) * (pre > 0.)
    gradients = {"a": rows.T @ hidden_derivative,
                 "b": hidden_derivative.sum(axis=0),
                 "c": hidden.T @ derivative,
                 "d": derivative.sum(axis=0)}
    return logmeanexp(q)-float(np.mean(scores[:nj])), gradients


def initialize(seed, dimensions=2, hidden=32):
    rng = np.random.default_rng(seed)
    return {"a": rng.normal(size=(dimensions,hidden))/math.sqrt(dimensions),
            "b": np.zeros(hidden),
            "c": rng.normal(size=(hidden,1))/math.sqrt(hidden),
            "d": np.zeros(1)}


def estimate_checkpoints(joint, reference, seed, checkpoints=(120,200)):
    assert joint.shape == reference.shape == (128,2)
    assert np.isfinite(joint).all() and np.isfinite(reference).all()
    pooled = np.concatenate((joint,reference))
    center, scale = pooled.mean(axis=0),pooled.std(axis=0)+1e-6
    j, q = (joint-center)/scale,(reference-center)/scale
    parameters = initialize(seed)
    means = {key: np.zeros_like(value) for key,value in parameters.items()}
    squares = {key: np.zeros_like(value) for key,value in parameters.items()}
    rng = np.random.default_rng(seed)
    output = {}
    for step in range(1,max(checkpoints)+1):
        jb = j[rng.integers(0,len(j),size=64)]
        qb = q[rng.integers(0,len(q),size=64)]
        _, gradients = objective_and_gradient(parameters,jb,qb)
        for key in parameters:
            grad = gradients[key]
            means[key] = .9*means[key]+.1*grad
            squares[key] = .999*squares[key]+.001*(grad*grad)
            m = means[key]/(1.-.9**step)
            v = squares[key]/(1.-.999**step)
            parameters[key] -= .001*m/(np.sqrt(v)+1e-8)
        if step in checkpoints:
            js,_,_ = forward(parameters,j)
            qs,_,_ = forward(parameters,q)
            output[step] = max(float(np.mean(js))-logmeanexp(qs),0.)
            if not math.isfinite(output[step]):
                raise ArithmeticError("nonfinite output; dataset will be recorded as failure")
    return output


def gradient_preflight():
    # Fixed tiny algebraic control, not outcome-driven estimator tuning.
    parameters = initialize(919,dimensions=2,hidden=3)
    joint = np.array([[-.7,.2],[.5,.8],[1.2,-.3]])
    reference = np.array([[-.7,.8],[.5,-.3],[1.2,.2]])
    _, gradients = objective_and_gradient(parameters,joint,reference)
    largest = 0.
    eps=1e-6
    for name, array in parameters.items():
        for index in np.ndindex(array.shape):
            initial=float(array[index])
            array[index]=initial+eps
            up,_=objective_and_gradient(parameters,joint,reference)
            array[index]=initial-eps
            down,_=objective_and_gradient(parameters,joint,reference)
            array[index]=initial
            largest=max(largest,abs((up-down)/(2*eps)-gradients[name][index]))
    if largest>1e-6:
        raise ArithmeticError(f"gradient preflight failed: {largest}")
    return largest


def gaussian_style_tail(observed, null_values):
    center = statistics.mean(null_values)
    sigma = math.sqrt(max(statistics.pvariance(null_values),1e-6))
    return .5*math.erfc((observed-center)/(sigma*math.sqrt(2.)))


def wilson(k,n):
    z=1.959963984540054
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [center-radius,center+radius]


def run():
    protocol_path=HERE/'experiment_protocol.json'
    protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if protocol_hash!=PROTOCOL_SHA256:
        raise RuntimeError("protocol changed; refusing to run")
    if (HERE/'neural_results.json').exists():
        raise RuntimeError("one-shot results already exist; refusing accidental rerun")
    start=time.perf_counter()
    preflight=gradient_preflight()
    records=[]
    failures=[]
    for rho in (0.,.9):
        for seed in range(1000,1020):
            try:
                rng=np.random.default_rng(seed)
                x,z=rng.normal(size=(2,128))
                y=rho*x+math.sqrt(1-rho*rho)*z
                joint=np.column_stack((x,y))
                permutations=np.random.default_rng(20000+seed)
                observed_q=np.column_stack((x,y[permutations.permutation(128)]))
                observed=estimate_checkpoints(joint,observed_q,42,(200,))[200]
                nulls={mode:{120:[],200:[]} for mode in ('retained_joint','permuted_joint')}
                reference_hashes=[]
                for rep in range(3):
                    shuffled_joint=np.column_stack((x,y[permutations.permutation(128)]))
                    shared_reference=np.column_stack((x,y[permutations.permutation(128)]))
                    refhash=hashlib.sha256(shared_reference.tobytes()).hexdigest()
                    for mode,p in [('retained_joint',joint),('permuted_joint',shuffled_joint)]:
                        assert hashlib.sha256(shared_reference.tobytes()).hexdigest()==refhash
                        estimates=estimate_checkpoints(p,shared_reference,100+rep)
                        for steps in (120,200):
                            nulls[mode][steps].append(estimates[steps])
                    reference_hashes.append(refhash)
                for mode in nulls:
                    for steps,nv in nulls[mode].items():
                        records.append({'rho':rho,'dataset_seed':seed,'construction':mode,'null_steps':steps,
                                        'observed_steps':200,'observed_mi':observed,'null_values':nv,
                                        'null_mean':statistics.mean(nv),
                                        'gaussian_style_tail_score':gaussian_style_tail(observed,nv),
                                        'rank_style_diagnostic':(1+sum(v>=observed for v in nv))/4,
                                        'matched_reference_sha256':reference_hashes})
                print(f'Completed rho={rho:g} dataset_seed={seed}',flush=True)
            except Exception as exc:
                failures.append({'rho':rho,'dataset_seed':seed,'type':type(exc).__name__,'message':str(exc)})
                print(f'FAILED rho={rho:g} seed={seed}: {type(exc).__name__}: {exc}',flush=True)
    summaries=[]
    for rho in (0.,.9):
        for mode in ('retained_joint','permuted_joint'):
            for steps in (120,200):
                subset=[r for r in records if (r['rho'],r['construction'],r['null_steps'])==(rho,mode,steps)]
                if not subset:
                    continue
                n=len(subset)
                k=sum(r['gaussian_style_tail_score']<=.05 for r in subset)
                summaries.append({'rho':rho,'construction':mode,'null_steps':steps,'independent_dataset_count':n,
                                  'mean_observed_mi':statistics.mean(r['observed_mi'] for r in subset),
                                  'mean_nominal_null_mi':statistics.mean(r['null_mean'] for r in subset),
                                  'mean_observed_minus_null':statistics.mean(r['observed_mi']-r['null_mean'] for r in subset),
                                  'secondary_gaussian_score_le_0_05_count':k,
                                  'secondary_gaussian_score_le_0_05_fraction':k/n,
                                  'secondary_fraction_wilson_95':wilson(k,n),
                                  'minimum_rank_style_diagnostic':min(r['rank_style_diagnostic'] for r in subset)})
    index={(r['rho'],r['dataset_seed'],r['construction'],r['null_steps']):r for r in records}
    contrasts=[]
    for rho in (0.,.9):
        for seed in range(1000,1020):
            try:
                r120=index[(rho,seed,'retained_joint',120)]['null_mean']
                r200=index[(rho,seed,'retained_joint',200)]['null_mean']
                p120=index[(rho,seed,'permuted_joint',120)]['null_mean']
                p200=index[(rho,seed,'permuted_joint',200)]['null_mean']
                contrasts.append({'rho':rho,'dataset_seed':seed,'retained_minus_permuted_at_120':r120-p120,
                                  'retained_minus_permuted_at_200':r200-p200,
                                  'budget_200_minus_120_retained':r200-r120,
                                  'budget_200_minus_120_permuted':p200-p120})
            except KeyError:
                continue
    contrast_summary=[]
    for rho in (0.,.9):
        subset=[r for r in contrasts if r['rho']==rho]
        if subset:
            contrast_summary.append({'rho':rho,'independent_dataset_count':len(subset),
                                     **{key:statistics.mean(r[key] for r in subset) for key in subset[0] if key not in ('rho','dataset_seed')}})
    result={'experiment':'precommitted independent toy standard-gradient mechanism demonstration',
            'not':'faithful upstream neural estimator reproduction or operational FPR estimate',
            'protocol_sha256':protocol_hash,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
                           'blas_threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
            'runtime_seconds':time.perf_counter()-start,'gradient_preflight_max_abs_error':preflight,
            'predeclared_dataset_count':40,'completed_dataset_count':len(records)//4,
            'model_fit_trajectories':len(records)//4*7,'failures':failures,
            'record_count':len(records),'all_recorded_outputs_finite':all(math.isfinite(r[k]) for r in records for k in ('observed_mi','null_mean','gaussian_style_tail_score','rank_style_diagnostic')),
            'primary_cell_summaries_and_secondary_tails':summaries,'primary_paired_contrast_summary':contrast_summary,
            'per_dataset_paired_contrasts':contrasts,'records':records,
            'scope_limits':['Gaussian-style tail scores with three nominal null draws are not validated p-values in either construction.',
                            'Rank-style diagnostic minimum is 0.25; B=3 cannot support a 0.05 permutation rejection.',
                            'Fixed differing observed/null network seeds mirror a source nuisance but break any claim this is an exact exchangeable randomization statistic.',
                            'Independent datasets, not the three null fits or pooled rho conditions, are the descriptive replication units.',
                            'Temporal/run/agent dependence, aggregation, empirical calibration, ROC, and union metrics are not tested.',
                            'Standard gradient normalization differs from source; these score fractions are NOT measured FPR/power of the published implementation.',
                            'No outcome-driven reruns or hyperparameter/seed changes. No external repository code imported/executed.']}
    (HERE/'neural_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (HERE/'neural_cell_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    print(json.dumps({'runtime_seconds':result['runtime_seconds'],'completed_dataset_count':result['completed_dataset_count'],
                      'failures':failures,'cell_summaries':summaries,'paired_contrasts':contrast_summary},indent=2),flush=True)

if __name__=='__main__':
    run()
