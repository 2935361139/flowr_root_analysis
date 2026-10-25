"""Small, dependency-light checks of attribution and selection semantics."""
import unittest
import numpy as np
from trace_core import GROUPS, PREFIX, build_plan, decompose, exact_shapley, group_mask, readout, status


class TraceChecks(unittest.TestCase):
    def test_known_additive_effects(self):
        effects = np.array([2., -3., 0., 5.])
        values = np.array([10+sum(effects[g] for g in range(4) if s & (1 << g)) for s in range(16)])
        np.testing.assert_allclose(exact_shapley(values), effects, atol=1e-12)

    def test_interaction_shared_and_null_groups(self):
        values = np.array([6.*bool(s & 1)*bool(s & 2) for s in range(16)])
        np.testing.assert_allclose(exact_shapley(values), [3., 3., 0., 0.], atol=1e-12)

    def test_directional_fixed_identity(self):
        m = group_mask(2, 9)
        np.testing.assert_array_equal(m[0], [0, 0, 1, 1, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(m[1], [1, 1, 0, 0, 2, 2, 3, 3, 3])

    def weights(self):
        r = np.random.default_rng(809)
        return dict(w0=r.normal(size=(7, 9)), b0=r.normal(size=7),
                    w2=r.normal(size=(6, 7)), b2=r.normal(size=6),
                    wp=r.normal(size=(5, 6)), bp=r.normal(size=5))

    def test_head_orientation_symmetry(self):
        x = np.random.default_rng(88).normal(size=(5, 2, 9))
        np.testing.assert_allclose(readout(x, self.weights()), readout(x[:, ::-1], self.weights()), atol=1e-12)

    def test_permutation_equivariance_of_pair_attribution(self):
        r = np.random.default_rng(99)
        a, b = r.normal(size=(5, 2, 9)), r.normal(size=(5, 2, 9))
        order = [4, 1, 0, 3, 2]
        first = decompose(a, b, self.weights(), 2)
        permuted = decompose(a[order], b[order], self.weights(), 2)
        np.testing.assert_allclose(first['order_phi'][:, order], permuted['order_phi'], atol=1e-12)

    def test_no_change_has_no_contribution(self):
        a = np.random.default_rng(22).normal(size=(4, 2, 9))
        value = decompose(a, a.copy(), self.weights(), 2)
        np.testing.assert_allclose(value['order_phi'], 0, atol=1e-12)
        np.testing.assert_allclose(value['margin_phi'], 0, atol=1e-12)

    def test_signed_changes_conserve_nonlinear_score(self):
        r = np.random.default_rng(20)
        a, b = r.normal(size=(3, 2, 9)), r.normal(size=(3, 2, 9))
        value = decompose(a, b, self.weights(), 2)
        np.testing.assert_allclose(value['order_phi'].sum(0), value['expected_order'][15]-value['expected_order'][0], atol=1e-12)

    def record(self, seed, site, values):
        event = PREFIX+str(site)
        return dict(report=dict(case='unseen_case', ligand_idx=99, seed=seed,
                                fixed_mapping={str(site): dict(reference_index=123)}),
                    frames=[dict(call=k, chemistry=dict(status='assessed' if v is not None else 'unbuildable'),
                                 events=[event] if v else []) for k, v in enumerate(values)])

    def test_unknown_is_not_negative(self):
        frame = dict(chemistry=dict(status='unbuildable'), events=[])
        self.assertIsNone(status(frame, PREFIX+'7'))
        plan = build_plan({'renamed': self.record(71, 7, [None, True, False])})
        self.assertEqual([(j['before'], j['after'], j['kind']) for j in plan['jobs']], [(1, 2, 'recovery')])
        self.assertTrue(any('Unknown' in s['reason'] for s in plan['skipped']))

    def test_generic_indices_and_same_reference_controls(self):
        records = {'different_failure': self.record(71, 7, [False, True, False, True]),
                   'different_control': self.record(12, 19, [False, False, False, False])}
        plan = build_plan(records)
        events = [j for j in plan['jobs'] if j['kind'] != 'same_input_no_event_control']
        self.assertEqual(len(events), 3)
        self.assertEqual([j['terminal_segment_onset'] for j in events], [False, False, True])
        controls = [j for j in plan['jobs'] if j['kind'] == 'same_input_no_event_control']
        self.assertTrue(all(j['site'] == 19 for j in controls))
        self.assertTrue(all(j['site'] == 7 for j in events))
        self.assertEqual(plan['captures']['different_failure'], [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
