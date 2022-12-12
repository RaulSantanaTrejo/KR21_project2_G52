import itertools
from typing import Union
import pandas as pd

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

    def d_separation(self, X:[str], Y:[str], Z:[str]):
        self.prune(X + Y, Z)
        all_combinations = itertools.product(X,Y)
        for (start, end) in all_combinations:
            if networkx.has_path(self.bn.structure, start,end):
                return False
        return True

    def variable_elimination(self, elimination_order: list, input_variable: string):

        for var in elimination_order:

            cpt = self.bn.get_cpt(var)

            row_t = cpt.loc[cpt[var] == True]
            row_f = cpt.loc[cpt[var] == False]

            # Access the value in column 'p' of the retrieved row
            value_t = row_t.iloc[0]['p']
            value_f = row_f.iloc[0]['p']

            # Convert the value to a float
            prob_t = pd.to_numeric(value_t, errors='coerce')
            prob_f = pd.to_numeric(value_f, errors='coerce')

            var_children = self.bn.get_children(var)

            for child in var_children:
                # get all the cpt's of the children
                cpt_child = self.bn.get_cpt(child)

                # Multiply the values in column 'p' where the corresponding value in column var is True by the float value
                cpt_child.loc[cpt_child[var] == True, 'p'] = cpt_child.loc[cpt_child[var] == True, 'p'] * prob_t

                # Multiply the values in column 'p' where the corresponding value in column var is False by the float value
                cpt_child.loc[cpt_child[var] == False, 'p'] = cpt_child.loc[cpt_child[var] == False, 'p'] * prob_f

                groups = cpt_child.columns.drop(var).drop('p').tolist()

                # add together the rows with the same values
                cpt_child = cpt_child.groupby(groups).sum().reset_index().drop(columns=[var])

                # update the CPT of the variable
                self.bn.update_cpt(child, cpt_child)

                cpt_2 = self.bn.get_cpt(child)

        return self.bn.get_cpt(input_variable)
                


# TODO: This is where your methods should go
