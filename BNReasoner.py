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

    def independent(self, X: [str], Y: [str], Z: [str]):
        return self.d_separation(X, Y, Z)

    def marginilization(self, f, x:str):
        Y = self.bn.get_all_variables().remove(x)
        original_cpt = self.bn.get_cpt(x)
        new_table_dict = {}
        for row_num, content in original_cpt.iterrows():
            key = ()
            if key in new_table_dict:
                new_table_dict[key] = new_table_dict[key] + content['p']
            else:
                new_table_dict[key] = content['p']

        for key,value in new_table_dict:
            vars = key.split("_")
            for var in vars:
                name_truth =  var.split("=")
                name = name_truth[0]
                truth = bool(name_truth[1])




        # TODO: This is where your methods should go
