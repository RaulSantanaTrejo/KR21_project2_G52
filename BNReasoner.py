import itertools
from typing import Union, List
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
    
    
     
    ### MAXING OUT ###
    
     def max_out(self, cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
        """returns a cpt which contains a new column called archive.
        This column is a record of the previous instances of the maxedout variables and it contains
        a list of touples
        """
        if variable not in cpt.columns:
            print("variable we are tryng to max out is not in cpt")
            return None

        if [variable] == [c for c in cpt.columns if c != 'p' and c != 'archive']:
            print("ERROR in trying to maxout: variable is alone in the cpt, cant be maxed out")
            exit()
            return None

        the_forbidden_list = [variable, 'p', 'archive']
        group_var = [x for x in cpt.columns if x not in the_forbidden_list]
        # print("grouping for: ",group_var)

        new_cpt = cpt.loc[cpt.groupby(group_var)['p'].idxmax()]  # BLESS
        # print("new cpt after groupby: \n",new_cpt)
        if "archive" not in new_cpt.columns:  # if it is the first time
            list_dict = []
            for i, row in new_cpt.iterrows():
                # list_dict.append({var:row[var]}) #First approach was with dictionaries
                list_dict.append([(variable, row[variable])])
            new_cpt.insert(0, "archive", list_dict)
        else:
            for i, row in new_cpt.iterrows():
                # archive_dict=row["archive"]
                # archive_dict[var]=row[var]
                # row["archive"]=archive_dict
                archive_list = row["archive"]
                archive_list.append((variable, row[variable]))
                row["archive"] = archive_list

        new_cpt = new_cpt.drop(variable, 1)

        return new_cpt
    
    
    ### FACTOR MULTIPLICATION ###
    
    def factor_multiplication(self, set_of_cpts: List[pd.DataFrame]) -> pd.DataFrame:
        """Input: a list of cpts 
           Output: result of factor moltiplication between the cpts"""

        vars = []
        
        for cpt in set_of_cpts:
            for column_header in list(cpt.columns)[:-1]:
                if column_header not in vars:
                    vars.append(column_header)
        worlds = [list(i) for i in itertools.product([True, False], repeat=len(vars))]
        instantions = []
        for world in worlds:
            evi = {}
            for i, v in enumerate(vars):
                evi[v] = world[i]
            instantions.append(pd.Series(evi))
        good_instantions = []
        for inst in instantions:
            s = 0
            for c in set_of_cpts:
                if not BayesNet.get_compatible_instantiations_table(inst, c).empty:
                    s += 1
            if s == len(set_of_cpts):
                good_instantions.append(inst)
        list_vars = [list(i.to_dict().keys()) for i in good_instantions]
        good_vars = []
        for i in list_vars:
            for v in i:
                if v not in good_vars:
                    good_vars.append(v)
        good_insts = [list(i.to_dict().values()) for i in good_instantions]
        good_insts.sort()
        good_insts = list(good_insts for good_insts, _ in
                          itertools.groupby(good_insts)) 
        result_cpt = pd.DataFrame(good_insts, columns=good_vars)
        result_cpt['p'] = 1
        for i in range(len(good_insts)):
            inst = pd.Series(result_cpt.iloc[i][:-1], good_vars)
            for current_cpt in set_of_cpts:
                right_row = BayesNet.get_compatible_instantiations_table(inst, current_cpt)
                result_cpt.loc[i, 'p'] *= right_row['p'].values[0]
        return result_cpt
    
    def multiply_many_factors(self, factors: List[pd.DataFrame]) -> pd.DataFrame:
        if len(factors) == 1:
            return factors[0]
        else:
            result = factors[0]
            for i in range(1, len(factors)):
                common_vars = result.columns.intersection(factors[i].columns).tolist()
                if 'p' in common_vars:
                    common_vars.remove('p')
                if "archive" in common_vars:
                    common_vars.remove("archive")
                if len(common_vars) > 0:
                    result = self.multiply_factors(result, factors[i], common_vars)
            return result

    def multiply_factors(self, factor1: pd.DataFrame, factor2: pd.DataFrame,
                         common_vars: List[str]) -> pd.DataFrame:
        merged = factor1.merge(factor2, on=common_vars, how='inner')
        merged = merged.assign(p=merged.p_x * merged.p_y, ).drop(columns=['p_x', 'p_y'])
        if "archive" in factor1.columns and "archive" in factor2.columns:
            merged = merged.assign(archive=merged.archive_x + merged.archive_y).drop(
                columns=['archive_x', 'archive_y'])
        return merged

   


# TODO: This is where your methods should go
