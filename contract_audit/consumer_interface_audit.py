#!/usr/bin/env python3
"""Original data-only consumer contract demo; never executes repository code.

This does not execute the proposed source patch, stress pipeline or a real ROC.
It shows how the verified consumer formula responds to a hypothetical p-value
quarantine while raw fields remain untouched.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path(os.environ.get('ADAPTIVE_SOURCE_PATH',str(Path(__file__).resolve().parent.parent / 'source' / 'run_adaptive_lowrate.source.txt')))


def source_contract():
    source=SOURCE.read_text()
    tree=ast.parse(source)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='UNION_COMPONENTS' for t in n.targets))
    components=ast.literal_eval(assignment.value)
    pvalue_dicts=[n for n in ast.walk(tree) if isinstance(n,ast.DictComp)
                 and ast.unparse(n.value)=="info['p_value']"]
    holm_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Name) and n.func.id=='_holm_step_down']
    return {'source_path':'source/'+SOURCE.name,'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'union_components':sorted(components),'union_line':assignment.lineno,
            'reported_p_dictionary_lines':[n.lineno for n in pvalue_dicts],
            'holm_call_lines':[n.lineno for n in holm_calls]}


def holm_rejections(pvalues,alpha):
    """Independently implemented standard Holm step-down from mathematical rule."""
    ordered=sorted(pvalues.items(),key=lambda item:item[1])
    rejected=[]
    for position,(name,value) in enumerate(ordered):
        if value>alpha/(len(ordered)-position):
            break
        rejected.append(name)
    return rejected


def quarantine_data(record):
    result=copy.deepcopy(record)
    old=result['p_value']
    result['p_value']=1.
    result['details'].update({'p_value_status':'unavailable','p_value_policy':'conservative_no_rejection',
                              'reference_kind':'same_joint_marginal_resampling','uncalibrated_tail_score':old})
    return result


def report():
    source=source_contract()
    old={name:.5 for name in source['union_components']}
    old['cross_run_mi']=1e-8
    new={**old,'cross_run_mi':1.}
    before={str(alpha):holm_rejections(old,alpha) for alpha in (.01,.001)}
    after={str(alpha):holm_rejections(new,alpha) for alpha in (.01,.001)}
    sample={'statistic':.6928117741870494,'p_value':1e-8,
            'details':{'mi_values':[.6928117741870494],'dv_values':[.6928117741870494]}}
    patched=quarantine_data(sample)
    return {'scope':'hypothetical data-only interface demonstration, not executed upstream patch/integration or ROC',
            'source':source,'illustrative_old_pvalues':old,'illustrative_quarantined_pvalues':new,
            'holm_before':before,'holm_after':after,
            'adaptive_payload_rate_example':{'initial':1.,'before':.55,'after':1.15,
                                             'meaning':'MI-only constructed trigger is removed; algorithm switches from decay to growth'},
            'raw_statistic_preserved':sample['statistic']==patched['statistic'],
            'raw_mi_values_preserved':sample['details']['mi_values']==patched['details']['mi_values'],
            'raw_threshold_decision_at_0_5_preserved':(sample['statistic']>.5)==(patched['statistic']>.5),
            'not_established':'No original detector or adaptive pipeline executed; source patch/RNG invariance is a separate static review. No real ROC run.'}

if __name__=='__main__':
    print(json.dumps(report(),indent=2,allow_nan=False))
