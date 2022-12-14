import itertools
from typing import Union

import networkx

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def prune(self, queries: [str], evidence:[str]):
        change = False
        for variable in self.bn.get_all_variables():
            # remove all leafs not in queries or evidence
            variable_children = self.bn.get_children(variable)
            if not variable_children and variable not in queries and variable not in evidence:
                change = True
                self.bn.del_var(variable)
            if variable in evidence:
                for child in variable_children:
                    change = True
                    self.bn.del_edge((variable, child))

        if change:
            return self.prune(queries, evidence)

    def d_separation(self, X: [str], Y: [str], Z :[str]):
        self.prune(X + Y, Z)
        all_combinations = itertools.product(X,Y)
        for (start, end) in all_combinations:
            if networkx.has_path(self.bn.structure, start,end):
                return False
        return True



        # TODO: This is where your methods should go
