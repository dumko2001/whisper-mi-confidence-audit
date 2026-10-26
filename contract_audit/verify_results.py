#!/usr/bin/env python3
"""Read-only verification of the completed, frozen toy experiment; no fitting."""
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
result = json.loads((root/'neural_results.json').read_text())
provenance = json.loads((root.parent/'provenance.json').read_text())
for key, name in [('protocol_sha256','experiment_protocol.json'),('script_sha256','toy_neural_experiment.py')]:
    entry = provenance['files']['contract_audit/'+name]
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual == entry['staged_sha256']
    assert result[key] in (entry['recorded_sha256'], actual)
assert result['failures'] == []
assert result['completed_dataset_count'] == 40
assert result['model_fit_trajectories'] == 280
records = result['records']
assert len(records) == 160
actual = {(r['rho'],r['dataset_seed'],r['construction'],r['null_steps']) for r in records}
expected = {(rho,seed,mode,steps) for rho in (0.,.9) for seed in range(1000,1020)
            for mode in ('retained_joint','permuted_joint') for steps in (120,200)}
assert actual == expected
assert len(actual) == len(records)
for rho in (0.,.9):
    for seed in range(1000,1020):
        cells = [r for r in records if (r['rho'],r['dataset_seed']) == (rho,seed)]
        assert len({r['observed_mi'] for r in cells}) == 1
        assert len({tuple(r['matched_reference_sha256']) for r in cells}) == 1
        for r in cells:
            assert len(r['null_values']) == 3
            assert all(math.isfinite(v) for v in r['null_values'])
            assert math.isclose(r['null_mean'],statistics.mean(r['null_values']),abs_tol=1e-15)
            assert r['rank_style_diagnostic'] >= .25
for summary in result['primary_cell_summaries_and_secondary_tails']:
    cells=[r for r in records if (r['rho'],r['construction'],r['null_steps']) ==
           (summary['rho'],summary['construction'],summary['null_steps'])]
    assert len(cells) == 20
    assert summary['secondary_gaussian_score_le_0_05_count'] == sum(r['gaussian_style_tail_score'] <= .05 for r in cells)
    assert math.isclose(summary['mean_observed_mi'],statistics.mean(r['observed_mi'] for r in cells),abs_tol=1e-15)
    assert math.isclose(summary['mean_nominal_null_mi'],statistics.mean(r['null_mean'] for r in cells),abs_tol=1e-15)
print('PASS: recorded-to-public protocol/script provenance hashes; complete 40-dataset × 4-cell grid; 280 fit trajectories; identical observed/reference controls; finite outputs; means and secondary counts recomputed from raw records.')
