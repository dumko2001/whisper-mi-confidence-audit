"""Static source and own-mock checks only; never import/execute upstream code."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
before = (ROOT.parent / "source" / "mi.original.txt").read_text()
after = (ROOT.parent / "source" / "mi.patched.txt").read_text()
old = ast.parse(before)
new = ast.parse(after)

def fn(tree):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_cross_run_mi")

restored = copy.deepcopy(new)
f = fn(restored)
# Remove only the inserted status/diagnostic-details block.
status_block = f.body[-2]
assert isinstance(status_block, ast.If)
assert ast.dump(status_block.test) == "Name(id='use_dv', ctx=Load())"
assert len(status_block.body) == 1
assert isinstance(status_block.body[0], ast.Expr)
assert isinstance(status_block.body[0].value, ast.Call)
status_call = status_block.body[0].value
assert isinstance(status_call.func, ast.Attribute)
assert ast.dump(status_call.func.value) == "Name(id='details', ctx=Load())"
assert status_call.func.attr == "update"
status_dict = status_call.args[0]
assert [key.value for key in status_dict.keys] == [
    "p_value_status", "p_value_policy", "reference_kind", "uncalibrated_tail_score"
]
assert [val.value for val in status_dict.values[:3]] == [
    "unavailable", "conservative_no_rejection", "same_joint_marginal_resampling"
]
assert ast.dump(status_dict.values[3]) == "Name(id='uncalibrated_tail_score', ctx=Load())"
del f.body[-2]
# Undo only the assigned-name change and inserted conservative value.
dv_branch = next(n for n in f.body if isinstance(n, ast.If) and ast.dump(n.test) == "Name(id='use_dv', ctx=Load())")
score_index = next(i for i, n in enumerate(dv_branch.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "uncalibrated_tail_score")
score = dv_branch.body[score_index]
score.targets[0].id = "p_value"
sentinel = dv_branch.body[score_index + 1]
assert ast.dump(sentinel) == "Assign(targets=[Name(id='p_value', ctx=Store())], value=Constant(value=1.0))"
del dv_branch.body[score_index + 1]
assert ast.dump(old, include_attributes=False) == ast.dump(restored, include_attributes=False)

def calls(tree, keep):
    return [ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Call) and keep(n)]

estimator_unchanged = calls(old, lambda n: isinstance(n.func, ast.Name) and n.func.id == "estimate_mi") == calls(new, lambda n: isinstance(n.func, ast.Name) and n.func.id == "estimate_mi")
rng_unchanged = calls(old, lambda n: isinstance(n.func, ast.Attribute) and n.func.attr in {"permutation", "shuffle"}) == calls(new, lambda n: isinstance(n.func, ast.Attribute) and n.func.attr in {"permutation", "shuffle"})
assert estimator_unchanged and rng_unchanged

# Independent mock of the proposed output contract, not upstream code execution.
def quarantine_mock(statistic, mi_values, tail_score):
    return {"statistic": statistic, "p_value": 1.0, "details": {
        "mi_values": mi_values,
        "p_value_status": "unavailable",
        "p_value_policy": "conservative_no_rejection",
        "reference_kind": "same_joint_marginal_resampling",
        "uncalibrated_tail_score": tail_score,
    }}

for score in [0.0, 1e-12, 0.001, 0.5, 1.0]:
    raw = [0.1, 0.7]
    result = quarantine_mock(0.4, raw, score)
    assert result["statistic"] == 0.4 and result["details"]["mi_values"] is raw
    assert result["p_value"] == 1.0 and result["details"]["uncalibrated_tail_score"] == score
    assert not any(result["p_value"] <= alpha for alpha in [0.05, 0.01, 0.001])
    assert json.loads(json.dumps(result)) == result

def diff(path):
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="a/" + path, tofile="b/" + path))

primary = diff("auditor/detectors/mi.py")
(ROOT / "quarantine-mi-p-value.patch").write_text(primary)
(ROOT / "quarantine-mi-p-value-with-release-mirror.patch").write_text(primary + diff("release/code/auditor/detectors/mi.py"))
report = {
    "source_commit": "48e6713d3e26b49dd94fa6645b4aeac6cb25a229",
    "source_path": "auditor/detectors/mi.py",
    "expected_git_blob_sha1": "7f7717d60282829657619b8768510cb2dedf99f2",
    "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
    "patched_sha256": hashlib.sha256(after.encode()).hexdigest(),
    "semantic_ast_restoration_equal": True,
    "estimator_call_ast_unchanged": estimator_unchanged,
    "rng_call_ast_unchanged": rng_unchanged,
    "independent_mock_tail_scores_checked": 5,
    "upstream_code_executed": False,
    "actual_integration_tested": False,
    "caveat": "Quarantines unsupported confidence only; does not construct a calibrated independence test."
}
blob = b"blob " + str(len(before.encode())).encode() + b"\0" + before.encode()
report["local_git_blob_sha1"] = hashlib.sha1(blob).hexdigest()
assert report["local_git_blob_sha1"] == report["expected_git_blob_sha1"], report
(ROOT / "patch_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
