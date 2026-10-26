#!/usr/bin/env python3
"""Original, stdlib-only contract audit; NEVER imports or runs external repository code.

This is not a reproduction of the neural estimator, operational detector, paper,
or its FPR. A finite critic family makes the tested null estimand transparent.
Read-only source AST inspection is deliberately separate from toy math.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PIN = "48e6713d3e26b49dd94fa6645b4aeac6cb25a229"
BASE_URL = f"https://github.com/Ostailor/Steganographic_Collusion/blob/{PIN}/"
DEFAULT_SOURCE = Path(os.environ.get("AUDIT_SOURCE_DIR", str(Path(__file__).resolve().parent.parent / "source")))
CRITIC_GRID = (-8., -4., -2., -1., 0., 1., 2., 4., 8.)

@dataclass(frozen=True)
class Unit:
    stratum: str
    unit_id: str
    x: tuple[int, ...]
    y: tuple[int, ...]

@dataclass(frozen=True)
class Row:
    stratum: str
    x_unit: str
    y_unit: str
    time: int
    x: int
    y: int

@dataclass(frozen=True)
class EstimatorInput:
    joint: tuple[Row, ...]
    marginal: tuple[Row, ...]
    steps: int
    label: str


def stable_log_mean_exp(values: Sequence[float], source_epsilon: bool = False) -> float:
    """Independent mathematical implementation; finite nonempty inputs only."""
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("requires nonempty, finite scores")
    largest = max(values)
    average = math.fsum(math.exp(x - largest) for x in values) / len(values)
    return largest + math.log(average + (1e-12 if source_epsilon else 0.))


def naive_log_mean_exp(values: Sequence[float]) -> float:
    return math.log(math.fsum(math.exp(x) for x in values) / len(values))


def dv_value(joint: Sequence[tuple[int, int]], marginal: Sequence[tuple[int, int]], strength: float) -> float:
    # Bounded equality critic, not a neural network or fit.
    joint_scores = [strength if x == y else 0. for x, y in joint]
    marginal_scores = [strength if x == y else 0. for x, y in marginal]
    return math.fsum(joint_scores) / len(joint_scores) - stable_log_mean_exp(marginal_scores)


def finite_critic_statistic(joint: Sequence[tuple[int, int]], marginal: Sequence[tuple[int, int]]) -> float:
    # Same fixed critic set in observed and null: no unequal training budget.
    return max(dv_value(joint, marginal, t) for t in CRITIC_GRID)


def empirical_mi(pairs: Sequence[tuple[int, int]]) -> float:
    n = len(pairs)
    xy, x, y = Counter(pairs), Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    return math.fsum((count/n) * math.log(count*n/(x[a]*y[b])) for (a,b), count in xy.items())


def toy_units() -> tuple[Unit, ...]:
    # Assumption of THIS TOY: each unit is an exchangeable independent cluster
    # within its stratum; each two-round trajectory must remain intact.
    return tuple(Unit(s, f"{s}{i}", (b, 1-b), (b, 1-b))
                 for s in ("A", "B") for i, b in enumerate((0, 0, 1, 1)))


def paired_rows(units: Sequence[Unit], y_assignment: Sequence[int]) -> tuple[Row, ...]:
    if sorted(y_assignment) != list(range(len(units))):
        raise ValueError("assignment must be a full bijection")
    rows = []
    for i, j in enumerate(y_assignment):
        u, v = units[i], units[j]
        if u.stratum != v.stratum or len(u.x) != len(v.y):
            raise ValueError("block exchangeability or trajectory length violation")
        rows.extend(Row(u.stratum, u.unit_id, v.unit_id, t, x, v.y[t])
                    for t, x in enumerate(u.x))
    return tuple(rows)


def grouped_permutations(units: Sequence[Unit]):
    groups = {}
    for i, unit in enumerate(units):
        groups.setdefault(unit.stratum, []).append(i)
    indices = list(groups.values())
    for chosen in itertools.product(*(itertools.permutations(g) for g in indices)):
        assignment = list(range(len(units)))
        for positions, perm in zip(indices, chosen):
            for i, j in zip(positions, perm):
                assignment[i] = j
        yield tuple(assignment)


def original_style_inputs(units: Sequence[Unit], permutations: Sequence[Sequence[int]], equal_budget: bool = False):
    """Source-pattern abstraction: joint is fixed; only reference is permuted.

    The real source permutes individual rows. Here permutations already preserve
    blocks, making the counterexample stronger: repairing exchangeability of the
    reference alone still does not change the nominal-null joint.
    """
    joint = paired_rows(units, tuple(range(len(units))))
    return [EstimatorInput(joint, paired_rows(units, p), 200 if equal_budget else 120,
                           "nominal_null_same_joint") for p in permutations]


def fixed_style_input(units: Sequence[Unit], joint_permutation: Sequence[int], reference_permutation: Sequence[int]):
    # Permute tested Y trajectories first, then permute this transformed Y again
    # for the product-reference draw. Both use the same 200-step contract.
    joint = paired_rows(units, joint_permutation)
    # Compose assignments to retain original Y-unit identity in metadata.
    composed = tuple(joint_permutation[j] for j in reference_permutation)
    marginal = paired_rows(units, composed)
    return EstimatorInput(joint, marginal, 200, "permuted_joint_null")


def values(rows: Sequence[Row]) -> tuple[tuple[int, int], ...]:
    return tuple((row.x, row.y) for row in rows)


def exact_product_at_round(units: Sequence[Unit]) -> tuple[tuple[int, int], ...]:
    # Equal-sized strata toy: all cross-unit X/Y pairs, within stratum and round.
    # This removes finite random product-reference noise, not observed/null noise.
    return tuple((u.x[t], v.y[t]) for u in units for v in units
                 if u.stratum == v.stratum for t in range(len(u.x)))


def source_contract(source_dir: Path) -> dict:
    paths = {"detector": source_dir / "auditor__detectors__mi.py",
             "estimator": source_dir / "auditor__mi_estimators.py"}
    texts = {key: path.read_text() for key, path in paths.items()}
    trees = {key: ast.parse(value) for key, value in texts.items()}
    estimate_calls = []
    for node in ast.walk(trees["detector"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "estimate_mi":
            kw = {p.arg: ast.literal_eval(p.value) if isinstance(p.value, ast.Constant) else ast.unparse(p.value)
                  for p in node.keywords}
            estimate_calls.append({"line": node.lineno, "joint_argument": ast.unparse(node.args[0]),
                                   "marginal_argument": ast.unparse(node.args[1]), **kw})
    estimate_calls.sort(key=lambda c: c["line"])
    helper = next(n for n in trees["estimator"].body
                  if isinstance(n, ast.FunctionDef) and n.name == "_log_mean_exp")
    helper_code = ast.unparse(helper)
    stable_exp = any(isinstance(n, ast.Call) and ast.unparse(n.func) == "np.exp"
                     and isinstance(n.args[0], ast.BinOp) and isinstance(n.args[0].op, ast.Sub)
                     and ast.unparse(n.args[0].right) == "max_val" for n in ast.walk(helper))
    final_uses_helper = "np.mean(f_joint) - _log_mean_exp(f_marg)" in texts["estimator"]
    return {"commit": PIN, "source_access": "read-only AST; external code never imported/executed",
            "source_files": {key: {"path": "source/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "url": BASE_URL + ("auditor/detectors/mi.py" if key == "detector" else "auditor/mi_estimators.py")}
                             for key, path in paths.items()},
            "estimate_mi_calls": estimate_calls, "dv_helper_max_shifted": stable_exp,
            "dv_final_uses_helper": final_uses_helper,
            "dv_helper_lines": [helper.lineno, helper.end_lineno]}


def build_report(source_dir: Path) -> dict:
    units = toy_units()
    perms = list(grouped_permutations(units))
    product = exact_product_at_round(units)
    observed = values(paired_rows(units, tuple(range(len(units)))))
    original = original_style_inputs(units, perms[:3])
    independent = ((0,0), (0,1), (1,0), (1,1))
    dependent = ((0,0), (0,0), (1,1), (1,1))
    # A nontrivial whole-trajectory permutation creates exact empirical independence.
    illustration = (0,2,1,3,4,6,5,7)
    fixed = fixed_style_input(units, illustration, illustration)
    null_stats = [finite_critic_statistic(values(paired_rows(units, p)), product) for p in perms]
    try:
        naive_log_mean_exp((1000., 999.))
        naive_overflow = False
    except OverflowError:
        naive_overflow = True
    src = source_contract(source_dir)
    return {"audit_type": "original bounded source-contract counterexample, not operational reproduction",
            "source": src,
            "null_contract": {
                "independent_plugin_mi": empirical_mi(independent),
                "dependent_plugin_mi": empirical_mi(dependent),
                "observed_dv_finite_critic": finite_critic_statistic(observed, product),
                "same_joint_nominal_null_dv_exact_product": [finite_critic_statistic(values(c.joint), product) for c in original],
                "true_independence_dv_fixed_critic_8": dv_value(independent, independent, 8.),
                "true_independence_optimized_over_same_finite_critic_grid": finite_critic_statistic(independent, independent),
                "fixed_illustrative_joint_plugin_mi": empirical_mi(values(fixed.joint)),
                "fixed_illustrative_joint_dv": finite_critic_statistic(values(fixed.joint), product),
                "source_same_joint_argument": all(c["joint_argument"] == "joint_sample" for c in src["estimate_mi_calls"]),
                "toy_nominal_null_same_joint_object": all(c.joint is original[0].joint for c in original),
                "source_observed_steps": 200, "source_nominal_null_steps": 120,
                "budget_ablation": "No neural fitting performed. Equalizing call metadata to 200 leaves joint dataflow unchanged.",
                "valid_toy_block_permutation_count": len(perms),
                "toy_null_statistics_all_finite": all(math.isfinite(x) for x in null_stats),
                "toy_null_statistic_range": [min(null_stats), max(null_stats)],
                "toy_null_statistic_mean": math.fsum(null_stats)/len(null_stats),
                "toy_null_preserves": ["Y-trajectory content and order", "stratum", "independent unit", "X rows", "feature marginals"],
                "toy_null_caveat": "Includes identity and residual associations as a valid finite permutation set should; not every draw yields zero MI."},
            "numerics": {"hypothesis_original_DV_direct_exp_overflows": "NOT SUPPORTED: pinned source already uses max-shifted log-mean-exp and stable softmax",
                         "generic_naive_exp_control_overflows_at_1000": naive_overflow,
                         "standard_stable_log_mean_exp_1000_999": stable_log_mean_exp((1000.,999.)),
                         "source_style_epsilon_stable_log_mean_exp_1000_999": stable_log_mean_exp((1000.,999.), source_epsilon=True),
                         "moderate_direct_stable_abs_difference": abs(naive_log_mean_exp((-2.,0.,2.)) - stable_log_mean_exp((-2.,0.,2.))),
                         "scope": "Finite constructed critic scores, not an end-to-end guarantee for training or extreme nonfinite network activations."},
            "limitations": ["No external repository code executed; no neural training, paid APIs, or LLM calls.",
                            "No operational p-value, FPR, detector power, benchmark effect, or headline-paper invalidation established.",
                            "Toy independent clusters and within-stratum exchangeability are explicit assumptions, not established for repository run/agent/round rows.",
                            "A production fix must define exchangeable/independent units and all conditioning, then use the same full estimation pipeline and budget for observed and null statistics.",
                            "The exact-product mathematical counterexample isolates the null estimand; it does not reproduce the random finite-reference or trained-network distributions.",
                            "Exported MI p-value versus metric union/ROC scope must be assessed separately from this two-file audit."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.source_dir)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")

if __name__ == "__main__":
    main()
