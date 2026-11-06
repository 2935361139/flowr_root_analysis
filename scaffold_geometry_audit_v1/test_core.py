import unittest
from core import atom_regions, fixed_pair_comparison


class RegionTests(unittest.TestCase):
    def test_atom_relabeling_does_not_change_regions(self):
        edges = [(0,1),(1,2),(2,3),(3,4)]
        a = atom_regions(range(5),edges,{0,1},{1})
        perm = {0:20,1:7,2:81,3:2,4:0}
        b = atom_regions(perm.values(),[(perm[i],perm[j]) for i,j in edges],{20,7},{7})
        self.assertEqual(a,{i:b[perm[i]] for i in a})
        self.assertEqual(a[4]['graph_hops_to_current_boundary'],2)

    def test_reference_anchor_not_forced_to_equal_output_attachment(self):
        roles = atom_regions(range(3),[(0,1),(0,2)],{0,1},{1})
        self.assertEqual(roles[1]['region'],'fixed_other')
        self.assertTrue(roles[1]['is_reference_attachment_atom'])
        self.assertFalse(roles[0]['is_reference_attachment_atom'])

    def test_disconnected_generated_component_is_not_called_near(self):
        roles = atom_regions(range(5),[(0,1),(1,2),(3,4)],{0},{0})
        self.assertIsNone(roles[4]['graph_hops_to_current_boundary'])

    def test_no_interface_is_unknown_distance_not_zero(self):
        self.assertIsNone(atom_regions([0,1],[],{0},{0})[1]['graph_hops_to_current_boundary'])

    def test_new_generated_atom_is_never_matched_to_reference_by_index(self):
        self.assertEqual(fixed_pair_comparison(8,10,{0:4},{(8,10)}),
                         'no_reference_atom_correspondence_for_generated_core')

    def test_fixed_comparison_requires_same_environment_atom(self):
        self.assertTrue(fixed_pair_comparison(0,10,{0:4},{(4,10)}).startswith('same_fixed'))
        self.assertTrue(fixed_pair_comparison(0,11,{0:4},{(4,10)}).startswith('fixed_atom'))

    def test_invalid_mapping_is_an_error_not_a_normal_label(self):
        with self.assertRaises(ValueError): atom_regions([0],[],{0},{3})


if __name__ == '__main__': unittest.main()
