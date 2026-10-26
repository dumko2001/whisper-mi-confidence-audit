#!/usr/bin/env python3
"""Read saved bounded regression output; do not import or run upstream code."""
import hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parent
r=json.loads((root/'integration_results.json').read_text())
assert r['status']=='PASS' and r['warnings']==[]
assert r['runtime_counters']=={'detector_calls':8,'spy_estimator_calls':30,'actual_estimator_calls':30,'actual_training_steps_total':4560}
assert r['identical_complete_spy_calls_across_three_run_pairs']
assert r['numeric_existing_details_and_statistic_exactly_equal']
assert r['static_contract']['matching_estimator_and_rng_call_count']==9
assert len(r['spy_call_trace'])==15
for item in r['source_manifest'].values():
    b=(root.parent/item['path']).read_bytes()
    assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
a,b=r['numeric_before'],r['numeric_after']
assert a['statistic']==b['statistic'] and math.isfinite(a['statistic'])
extra={'p_value_status','p_value_policy','reference_kind','uncalibrated_tail_score'}
assert a['details']=={k:v for k,v in b['details'].items() if k not in extra}
assert b['p_value']==1. and b['details']['uncalibrated_tail_score']==a['p_value']
assert b['details']['p_value_status']=='unavailable'
assert b['details']['p_value_policy']=='conservative_no_rejection'
assert b['details']['reference_kind']=='same_joint_marginal_resampling'
assert r['spy_holm_before']==['cross_run_mi'] and r['spy_holm_after']==[]
for edge in r['zero_fit_edgecases'].values():
    assert edge['exactly_equal'] and edge['before']==edge['after']
print('PASS: saved synthetic regression counters, pinned hashes, exact preservation, quarantine fields, and zero-fit edgecases; no fitting or upstream execution.')
