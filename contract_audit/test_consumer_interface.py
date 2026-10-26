import unittest
import consumer_interface_audit as c

class ConsumerTests(unittest.TestCase):
    def test_readonly_source_includes_mi_and_consumes_reported_p(self):
        source=c.source_contract()
        self.assertIn('cross_run_mi',source['union_components'])
        self.assertEqual(len(source['union_components']),5)
        self.assertEqual(source['reported_p_dictionary_lines'],[81])
        self.assertEqual(source['holm_call_lines'],[82])

    def test_mi_only_rejection_removed_at_both_declared_alphas(self):
        result=c.report()
        for alpha in ('0.01','0.001'):
            self.assertEqual(result['holm_before'][alpha],['cross_run_mi'])
            self.assertEqual(result['holm_after'][alpha],[])

    def test_other_strong_signal_not_suppressed(self):
        values={name:.5 for name in c.source_contract()['union_components']}
        values['cross_run_mi']=1.
        values['watermark_variance']=1e-8
        self.assertEqual(c.holm_rejections(values,.01),['watermark_variance'])

    def test_raw_statistics_values_and_order_preserved_in_interface(self):
        records=[{'statistic':v,'p_value':1e-8,'details':{'mi_values':[v]}} for v in (.7,.1,.9,.4)]
        quarantined=[c.quarantine_data(r) for r in records]
        self.assertEqual([r['statistic'] for r in records],[r['statistic'] for r in quarantined])
        self.assertEqual([r['details']['mi_values'] for r in records],[r['details']['mi_values'] for r in quarantined])
        self.assertEqual(sorted(range(4),key=lambda i:records[i]['statistic']),
                         sorted(range(4),key=lambda i:quarantined[i]['statistic']))
        self.assertTrue(all(r['p_value']==1. for r in quarantined))
        self.assertTrue(all(r['p_value']==1e-8 for r in records))

if __name__=='__main__':
    unittest.main(verbosity=2)
