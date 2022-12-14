import unittest

from BNReasoner import BNReasoner


class MyTestCase(unittest.TestCase):
    def test_prune_1(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        reasoner.prune(['Winter?', 'Sprinkler?'], ['Rain?'])
        self.assertListEqual(reasoner.bn.get_all_variables(), ['Winter?', 'Sprinkler?', 'Rain?'])

    def test_prune_2(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        reasoner.prune(['Winter?', 'Sprinkler?', 'Wet Grass?'], ['Rain?'])
        self.assertListEqual(reasoner.bn.get_all_variables(), ['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?'])
        self.assertListEqual(reasoner.bn.get_children('Rain?'), [])

    def test_prune_3(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        reasoner.prune([], ['Winter?'])
        self.assertListEqual(reasoner.bn.get_all_variables(), ['Winter?'])

    def test_d_separation_1(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        self.assertTrue(reasoner.d_separation(['Winter?', 'Sprinkler?'], ['Slippery Road?'], ['Rain?']))

    def test_d_separation_2(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        self.assertFalse(reasoner.d_separation(['Winter?'], ['Sprinkler?'], ['Rain?']))

    def test_marginalization_1(self):
        reasoner = BNReasoner('testing/lecture_example.BIFXML')
        f = reasoner.bn.get_cpt('Slippery Road?')
        new_cpt = reasoner.marginalization(f, 'Rain?')
        self.assertEquals(new_cpt.iloc[0]['p'], 1.3)
        self.assertEquals(new_cpt.iloc[1]['p'], 0.7)


if __name__ == '__main__':
    unittest.main()
