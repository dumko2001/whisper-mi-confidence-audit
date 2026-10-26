"""Regression tests of the original harness and read-only source contracts."""
import math
import unittest
from collections import Counter
import contract_audit as a

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = a.toy_units()
        cls.identity = tuple(range(len(cls.units)))
        cls.perms = list(a.grouped_permutations(cls.units))
        cls.product = a.exact_product_at_round(cls.units)
        cls.illustration = (0,2,1,3,4,6,5,7)

    def test_source_contract_same_joint_argument_and_distinct_marginal(self):
        calls = a.source_contract(a.DEFAULT_SOURCE)["estimate_mi_calls"]
        self.assertEqual([(c["method"], c["steps"], c["joint_argument"], c["marginal_argument"]) for c in calls],
                         [("dv",200,"joint_sample","marg_sample"),
                          ("nwj",200,"joint_sample","marg_sample"),
                          ("dv",120,"joint_sample","marg_rep")])

    def test_source_dv_is_already_stabilized_rejects_overflow_hypothesis(self):
        src = a.source_contract(a.DEFAULT_SOURCE)
        self.assertTrue(src["dv_helper_max_shifted"])
        self.assertTrue(src["dv_final_uses_helper"])

    def test_original_style_retains_joint_identity_while_marginal_changes(self):
        calls = a.original_style_inputs(self.units, [self.identity, self.illustration])
        self.assertIs(calls[0].joint, calls[1].joint)
        self.assertNotEqual(calls[0].marginal, calls[1].marginal)
        self.assertTrue(all(r.x_unit == r.y_unit for r in calls[1].joint))

    def test_budget_asymmetry_is_separate_from_dataflow_defect(self):
        low = a.original_style_inputs(self.units, [self.illustration])
        equal = a.original_style_inputs(self.units, [self.illustration], equal_budget=True)
        self.assertEqual((low[0].steps, equal[0].steps), (120, 200))
        self.assertEqual(low[0].joint, equal[0].joint)
        self.assertEqual(low[0].marginal, equal[0].marginal)

    def test_fixed_joint_changes_tested_pairing_and_preserves_units(self):
        fixed = a.fixed_style_input(self.units, self.illustration, self.illustration)
        observed = a.paired_rows(self.units, self.identity)
        self.assertNotEqual(fixed.joint, observed)
        self.assertEqual(Counter((r.stratum,r.y_unit,r.time,r.y) for r in fixed.joint),
                         Counter((r.stratum,r.y_unit,r.time,r.y) for r in observed))
        self.assertEqual([(r.x_unit,r.time,r.x) for r in fixed.joint],
                         [(r.x_unit,r.time,r.x) for r in observed])
        self.assertEqual(fixed.steps, 200)
        for uid in {r.x_unit for r in fixed.joint}:
            block = [r for r in fixed.joint if r.x_unit == uid]
            self.assertEqual(len({r.y_unit for r in block}),1)
            self.assertEqual([r.time for r in block],[0,1])
            self.assertTrue(all(r.stratum == r.y_unit[0] for r in block))

    def test_known_independent_and_dependent_toys(self):
        independent = ((0,0),(0,1),(1,0),(1,1))
        dependent = ((0,0),(0,0),(1,1),(1,1))
        self.assertAlmostEqual(a.empirical_mi(independent),0.,places=14)
        self.assertAlmostEqual(a.empirical_mi(dependent),math.log(2.),places=14)
        self.assertAlmostEqual(a.finite_critic_statistic(independent,independent),0.,places=14)
        expected = math.log(2.) - math.log1p(math.exp(-8.))
        self.assertAlmostEqual(a.finite_critic_statistic(dependent,independent),expected,places=13)

    def test_same_joint_population_null_targets_observed_signal(self):
        calls = a.original_style_inputs(self.units, self.perms[:3], equal_budget=True)
        scores = [a.finite_critic_statistic(a.values(c.joint),self.product) for c in calls]
        self.assertTrue(all(score == scores[0] for score in scores))
        self.assertGreater(scores[0], .69)
        fixed = a.fixed_style_input(self.units,self.illustration,self.illustration)
        self.assertAlmostEqual(a.empirical_mi(a.values(fixed.joint)),0.,places=14)
        self.assertAlmostEqual(a.finite_critic_statistic(a.values(fixed.joint),self.product),0.,places=14)

    def test_exact_block_permutation_set_preserves_design_and_is_finite(self):
        self.assertEqual(len(self.perms),math.factorial(4)**2)
        self.assertEqual(len(set(self.perms)),len(self.perms))
        self.assertIn(self.identity,self.perms)
        observed = a.paired_rows(self.units,self.identity)
        original_y = Counter((r.stratum,r.y_unit,r.time,r.y) for r in observed)
        for perm in self.perms:
            rows = a.paired_rows(self.units,perm)
            self.assertEqual(Counter((r.stratum,r.y_unit,r.time,r.y) for r in rows),original_y)
            self.assertTrue(math.isfinite(a.finite_critic_statistic(a.values(rows),self.product)))

    def test_nonexchangeable_cross_stratum_permutation_rejected(self):
        invalid = (4,1,2,3,0,5,6,7)
        with self.assertRaises(ValueError):
            a.paired_rows(self.units,invalid)

    def test_stable_and_naive_agree_in_moderate_range(self):
        for scores in ((-2.,0.,2.), (1.,1.,1.), (0.,), (-30.,-25.,-20.)):
            self.assertAlmostEqual(a.naive_log_mean_exp(scores),a.stable_log_mean_exp(scores),places=13)

    def test_generic_naive_overflow_control_but_source_style_stays_finite(self):
        with self.assertRaises(OverflowError):
            a.naive_log_mean_exp((1000.,999.))
        result = a.stable_log_mean_exp((1000.,999.),source_epsilon=True)
        self.assertTrue(math.isfinite(result))
        self.assertAlmostEqual(result,999.6201145069583,places=10)

    def test_shift_invariance_of_dv_and_finite_score_domain(self):
        j,m = (8.,8.),(8.,0.)
        baseline = sum(j)/len(j)-a.stable_log_mean_exp(m,source_epsilon=True)
        shifted = sum(x+1000. for x in j)/len(j)-a.stable_log_mean_exp(tuple(x+1000. for x in m),source_epsilon=True)
        self.assertAlmostEqual(baseline,shifted,places=11)
        for bad in ((), (math.inf,), (math.nan,)):
            with self.assertRaises(ValueError):
                a.stable_log_mean_exp(bad)

if __name__ == "__main__":
    unittest.main(verbosity=2)
