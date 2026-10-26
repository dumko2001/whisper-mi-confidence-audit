#!/usr/bin/env python3
"""Bounded synthetic detector regression; static-only by default.

Only --execute with a fresh output directory runs the three hash-verified source
files. The numerical fixture, counters, and 90-second budget are preserved.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import signal
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCES={
    'original':(ROOT.parent/'source'/'mi.original.txt','55007e97dbdff88019a6b54b1349e88e383fc3a1e5d4d9f11a15e555b920cec4'),
    'patched':(ROOT.parent/'source'/'mi.patched.txt','27b9af5f8f44a923cbc2560a3510e2948bf2a2d44c9d03a2c056472b327d9afd'),
    'estimator':(ROOT.parent/'source'/'mi_estimators.py','36980b9f08a6160b6e9e82932e0f3e1bd126a01c1476d1d50f18f4b2a9fa84df')}

@dataclass
class DetectorResult:
    name: str
    statistic: float
    p_value: float
    details: dict


def read_verified_sources():
    manifest={}
    buffers={}
    for role,(path,expected) in SOURCES.items():
        raw=path.read_bytes()
        actual=hashlib.sha256(raw).hexdigest()
        if expected is not None and actual!=expected:
            raise RuntimeError(f'Unexpected source bytes: {role}')
        manifest[role]={'path':'source/'+path.name,'sha256':actual,'bytes':len(raw)}
        buffers[role]=raw
    return manifest,buffers


def static_contract(buffers):
    def selected_calls(raw):
        tree=ast.parse(raw)
        calls=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=ast.unparse(node.func)
                if name=='estimate_mi' or name.startswith(('np_rng.','rng.','np.random.','random.')):
                    calls.append((node.lineno,ast.dump(node,include_attributes=False)))
        return [value for _,value in sorted(calls)]
    before=selected_calls(buffers['original'])
    after=selected_calls(buffers['patched'])
    if before!=after:
        raise RuntimeError('Patch changes estimator/RNG call expressions')
    return {'matching_estimator_and_rng_call_count':len(before)}


def load_verified_module(name,role,buffers):
    # Execute only the immutable bytes checked against the hard-coded whitelist.
    # No SourceFileLoader, import lookup or stale .pyc cache participates.
    module=types.ModuleType(name)
    module.__file__=str(SOURCES[role][0])
    module.__package__=name.rpartition('.')[0]
    sys.modules[name]=module
    compiled=compile(buffers[role],module.__file__,'exec')
    exec(compiled,module.__dict__)
    return module


def install_local_adapters():
    for name in ('auditor','auditor.detectors'):
        package=types.ModuleType(name)
        package.__path__=[]
        sys.modules[name]=package
    base=types.ModuleType('auditor.detectors.base')
    base.DetectorContext=types.SimpleNamespace
    base.DetectorResult=DetectorResult
    sys.modules[base.__name__]=base
    features=types.ModuleType('auditor.detectors.features')
    class FeatureExtractor:
        def __init__(self,summary):
            self.summary=summary
        def agent_features(self,run):
            return self.summary.features[run.name]
    features.FeatureExtractor=FeatureExtractor
    sys.modules[features.__name__]=features
    pvalues=types.ModuleType('auditor.pvalues')
    def gaussian_p_value(value,mean,std,tail):
        if tail!='right':
            raise ValueError('Fixture supports only observed right tail')
        return .5*math.erfc((value-mean)/(std*math.sqrt(2.)))
    pvalues.gaussian_p_value=gaussian_p_value
    sys.modules[pvalues.__name__]=pvalues


def fixture():
    runs=[types.SimpleNamespace(name=f'run{i}') for i in range(3)]
    features={}
    for run_index,run in enumerate(runs):
        features[run.name]={}
        for agent in range(2):
            rounds=[]
            for t in range(8):
                bit=(t+agent)%2
                phase=(t+2*agent+run_index)%3
                rounds.append(types.SimpleNamespace(
                    token_count=20+3*t+agent+run_index,
                    avg_token_length=3.+.2*bit+.03*run_index,
                    uppercase_ratio=.04*phase,digit_ratio=.02*(t%3),
                    punctuation_ratio=.03*(1+bit),green_fraction=.3+.05*bit,
                    message_entropy=1.+.1*t+.02*run_index,
                    price_value=10.+t+agent+.5*run_index,
                    bid_value=8.+.7*t+.2*agent,
                    vote_accept=bool(bit)))
            features[run.name][f'agent{agent}']=types.SimpleNamespace(round_features=rounds)
    return types.SimpleNamespace(config=types.SimpleNamespace(name='cross_run_mi'),
                                 summary=types.SimpleNamespace(runs=runs,features=features))


def comparable(result):
    return {'statistic':result.statistic,'details':{k:v for k,v in result.details.items()
            if k not in ('p_value_status','p_value_policy','reference_kind','uncalibrated_tail_score')}}


def holm(pvalues,alpha):
    result=[]
    for i,(name,value) in enumerate(sorted(pvalues.items(),key=lambda item:item[1])):
        if value>alpha/(len(pvalues)-i):
            break
        result.append(name)
    return result


def execute_once():
    # This function is never invoked by the default path or during preparation.
    out=ROOT/'integration_results.json'
    started=ROOT/'integration_started.json'
    failure=ROOT/'integration_failure.json'
    if out.exists() or started.exists() or failure.exists():
        raise RuntimeError('A one-shot start/result/failure marker exists; refusing rerun')
    with started.open('x') as handle:
        json.dump({'started_unix_seconds':time.time()},handle)
    # Enforce before the first NumPy import in this standalone process.
    if 'numpy' in sys.modules:
        raise RuntimeError('NumPy was already imported before BLAS limits; refusing run')
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[variable]='1'
    import numpy as np
    start=time.perf_counter()
    manifest,buffers=read_verified_sources()
    static=static_contract(buffers)
    counters={'detector_calls':0,'spy_estimator_calls':0,'actual_estimator_calls':0,'actual_training_steps_total':0}
    install_local_adapters()
    estimator=load_verified_module('auditor.mi_estimators','estimator',buffers)
    before=load_verified_module('auditor.detectors.original_mi_audit','original',buffers)
    after=load_verified_module('auditor.detectors.patched_mi_audit','patched',buffers)
    traces=[]
    spy_results=[]
    for module in (before,after):
        trace=[]
        def spy(joint,marginal,**kw):
            counters['spy_estimator_calls']+=1
            trace.append({'joint_shape':list(joint.shape),'marginal_shape':list(marginal.shape),
                          'joint_sha256':hashlib.sha256(joint.tobytes()).hexdigest(),
                          'marginal_sha256':hashlib.sha256(marginal.tobytes()).hexdigest(),**kw})
            return .6 if kw['steps']==200 else .01
        module.estimate_mi=spy
        counters['detector_calls']+=1
        spy_results.append(module.run_cross_run_mi(fixture()))
        traces.append(trace)
    assert len(traces[0])==len(traces[1])==15
    assert traces[0]==traces[1]
    assert comparable(spy_results[0])==comparable(spy_results[1])
    assert spy_results[1].p_value==1.
    assert spy_results[0].p_value==spy_results[1].details['uncalibrated_tail_score']
    assert spy_results[1].details['p_value_status']=='unavailable'
    assert spy_results[1].details['p_value_policy']=='conservative_no_rejection'
    assert spy_results[1].details['reference_kind']=='same_joint_marginal_resampling'
    spy_ps={name:.5 for name in ('cross_run_mi','permutation_invariance','watermark_variance','acceptance_bias','signal_bias')}
    spy_ps['cross_run_mi']=spy_results[0].p_value
    old_rejections=holm(spy_ps,.01)
    spy_ps['cross_run_mi']=spy_results[1].p_value
    new_rejections=holm(spy_ps,.01)
    assert old_rejections==['cross_run_mi'] and new_rejections==[]
    edgecases={}
    for label in ('insufficient_runs','no_pairs'):
        pair=[]
        for module in (before,after):
            context=fixture()
            if label=='insufficient_runs':
                context.summary.runs=context.summary.runs[:1]
            else:
                context.summary.features={run.name:{} for run in context.summary.runs}
            counters['detector_calls']+=1
            pair.append(module.run_cross_run_mi(context))
        assert vars(pair[0])==vars(pair[1])
        assert pair[0].statistic==0. and pair[0].p_value==1.
        assert pair[0].details['reason']==label
        edgecases[label]={'before':vars(pair[0]),'after':vars(pair[1]),'exactly_equal':True}
    numerics=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        def counted_estimator(joint,marginal,**kw):
            counters['actual_estimator_calls']+=1
            counters['actual_training_steps_total']+=kw['steps']
            if counters['actual_estimator_calls']>30 or counters['actual_training_steps_total']>4560:
                raise RuntimeError('Precommitted actual-fit/step budget exceeded')
            return estimator.estimate_mi(joint,marginal,**kw)
        for module in (before,after):
            module.estimate_mi=counted_estimator
            counters['detector_calls']+=1
            numerics.append(module.run_cross_run_mi(fixture()))
    assert comparable(numerics[0])==comparable(numerics[1])
    assert numerics[1].p_value==1.
    assert numerics[1].details['p_value_status']=='unavailable'
    assert numerics[1].details['p_value_policy']=='conservative_no_rejection'
    assert numerics[1].details['reference_kind']=='same_joint_marginal_resampling'
    assert numerics[0].p_value==numerics[1].details['uncalibrated_tail_score']
    assert counters=={'detector_calls':8,'spy_estimator_calls':30,'actual_estimator_calls':30,'actual_training_steps_total':4560}
    for result in numerics:
        assert math.isfinite(result.statistic) and math.isfinite(result.p_value)
        for key in ('mi_values','dv_values','nwj_values','plugin_values'):
            assert all(math.isfinite(v) for v in result.details[key])
    payload={'status':'PASS',
             'scope':'actual inspected detector/estimator modules with explicitly local feature/container/Gaussian adapters; not full repository pipeline',
             'source_manifest':manifest,'static_contract':static,'runtime_seconds':time.perf_counter()-start,
             'runtime_counters':counters,'blas_thread_environment':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
             'identical_complete_spy_calls_across_three_run_pairs':traces[0]==traces[1],
             'spy_call_trace':traces[0],
             'spy_holm_before':old_rejections,'spy_holm_after':new_rejections,'zero_fit_edgecases':edgecases,
             'numeric_existing_details_and_statistic_exactly_equal':comparable(numerics[0])==comparable(numerics[1]),
             'numeric_before':vars(numerics[0]),'numeric_after':vars(numerics[1]),
             'warnings':[str(w.message) for w in caught],
             'limitations':['Local feature and Gaussian adapters, not full feature pipeline or upstream pvalues helper.',
                            'No real-data FPR, power, calibration, ROC or adaptive-run claim.',
                            'Passing a quarantine integration does not establish a valid independence-null p-value.']}
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return payload


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    manifest,buffers=read_verified_sources()
    static=static_contract(buffers)
    if not args.execute:
        print(json.dumps({'status':'STATIC_ONLY_NOT_EXECUTED','source_manifest':manifest,
                          'static_contract':static,'protocol':'contract_audit/integration_protocol.json'},indent=2))
        return
    if args.output_dir is None:
        parser.error('--execute requires --output-dir with a fresh empty location')
    output=args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError('Output directory must not exist or must be completely empty')
    output.mkdir(parents=True,exist_ok=True)
    protocol=(ROOT/'integration_protocol.json').read_bytes()
    ROOT=output
    (ROOT/'integration_protocol.json').write_bytes(protocol)
    def timeout_handler(signum,frame):
        raise TimeoutError('90-second precommitted bound reached')
    signal.signal(signal.SIGALRM,timeout_handler)
    signal.alarm(90)
    try:
        result=execute_once()
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        failure={'status':'FAIL','type':type(exc).__name__,'message':str(exc),
                 'no_automatic_retry':True}
        try:
            with (ROOT/'integration_failure.json').open('x') as handle:
                handle.write(json.dumps(failure,indent=2)+'\n')
        except FileExistsError:
            pass
        raise
    finally:
        signal.alarm(0)

if __name__=='__main__':
    main()
