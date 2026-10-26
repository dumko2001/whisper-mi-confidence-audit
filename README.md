# Neural-MI confidence-contract audit

Companion to **Auditing the exported p value of a neural information detector** by Sidharth Rajmohan, Independent researcher, India.

This repository audits an exported confidence value in the **Audit the Whisper collusion detector**. Its subject is multi-agent collusion detection. It contains exact constructed counterexamples, a separate synthetic neural mechanism experiment, an output-only quarantine proposal, and a bounded synthetic detector/estimator regression. A calibrated replacement test, operational false-positive rates, full-pipeline behavior, and effects on the original paper's headline results remain unassessed.

Repository: https://github.com/dumko2001/whisper-mi-confidence-audit

## Contents

- `paper/`: manuscript and PDF.
- `contract_audit/`: original tests and synthetic scripts, scientific protocols, saved numerical outputs, and read-only verifiers.
- `source/`: selected upstream MIT-licensed source at commit `48e6713d3e26b49dd94fa6645b4aeac6cb25a229`, plus the proposed output-only detector edit. Duplicate filename variants preserve recorded source-byte fingerprints; the detector alias has one extra terminal newline.
- `patches/`: proposed patches and an AST/own-mock verifier. Upstream adoption is unconfirmed.
- `environment.json`, `provenance.json`, `MANIFEST.json`: recorded environment, recorded-to-portable artifact fingerprints, and SHA-256 inventory.

All numerical data come from synthetic fixtures. Saved neural results retain their execution-time protocol/script hashes. `provenance.json` maps those hashes to the portable versions; numerical results are unchanged.

## Safe verification: no fitting or upstream execution

Use Python 3.12; NumPy 2.3.5 matches the recorded numerical environment. From a fresh working copy of the repository root:

```sh
python3 -B -m unittest discover -s contract_audit -p 'test_*.py' -v
python3 -B contract_audit/verify_results.py
python3 -B contract_audit/verify_integration_results.py
python3 -B contract_audit/integration_regression.py
```

The integration script's default is static-only. It reads pinned source bytes and checks AST call preservation without importing or executing the upstream modules. The contract and consumer tests likewise parse source only and execute original local mathematics/mock logic.

The patch verifier writes derived patches and `patch_validation.json`; use an expendable fresh copy:

```sh
python3 -B patches/verify_quarantine_patch.py
```

## Optional fresh reproduction

These commands run new experiments. Use a POSIX/Unix environment for the integration alarm, NumPy 2.3.5, and one BLAS thread. GNU `timeout` provides an additional outer process bound.

For the independent local mechanism experiment, copy only `toy_neural_experiment.py` and `experiment_protocol.json` into a new directory, then run there:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -B toy_neural_experiment.py
```

It refuses an existing `neural_results.json`. Preserve existing results and use a fresh directory for each run. This original NumPy model uses two-dimensional inputs and standard gradients. Its architecture and computation differ from the upstream estimator; published performance remains unassessed.

The bounded regression runs only the three hash-verified detector/estimator source files with local synthetic feature/container/Gaussian adapters:

```sh
timeout --signal=TERM --kill-after=5s 95s \
  python3 -B contract_audit/integration_regression.py \
  --execute --output-dir /chosen/new-empty-output-directory
```

Execution requires the explicit flag and a fresh empty directory. The harness preserves its internal 90-second alarm, exclusive one-shot markers, one-thread BLAS, eight detector invocations, 30 actual estimator calls, and 4,560-step bound. The regression uses installed local dependencies and the selected source modules. A failure is terminal for that output directory; preserve its markers.

## Interpretation

Resampling the product reference leaves dependence in the tested joint intact. Gaussian-tail calibration requires separate validation even with equal budgets or a permuted joint. The patch preserves diagnostic values while setting the compatibility sentinel `p_value=1.0` and explicit unavailable-inference metadata. The sentinel denotes unavailable inference; honesty and independence remain undetermined. It can change a combiner's result. A local `erfc` adapter computes the regression's tiny tail. The upstream p-value helper remains untested. Actual feature extraction, operational calibration, and complete downstream behavior remain untested here.

## Next steps

1. Quarantine the uncalibrated tail as a diagnostic score and propagate unavailable-inference status through exporters and combiners without discarding descriptive values.
2. Independently define an independence null and exchangeable units appropriate to the run/agent/round structure. Match observed/reference fitting, budgets, and randomization, with enough reference draws for the target significance level.
3. Before restoring confidence claims, assess calibration on held-out null cases and power on controlled alternatives, then test the complete feature, aggregation, serialization, and downstream consumer paths.

These proposed steps define the validation needed before confidence claims can be restored.

## Attribution and licensing

Original software and its documentation are licensed under MIT; see `LICENSE`. Copyright (c) 2026 Sidharth Rajmohan. The manuscript is excluded from the MIT grant. The upstream authors retain authorship of their work. `LICENSE.upstream-MIT.txt` and `source/LICENSE` preserve the MIT notice for copied upstream material. The proposed detector edit is distinguished from upstream.

## AI usage

AI tools were used for literature and code review, implementation, analysis, and manuscript drafting and editing.
