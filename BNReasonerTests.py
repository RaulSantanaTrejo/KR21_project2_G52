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


if __name__ == '__main__':
    unittest.main()
