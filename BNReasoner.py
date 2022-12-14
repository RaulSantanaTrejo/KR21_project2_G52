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

    def variable_elimination(self, elimination_order: list, input_variable: str):

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
    
    
    def factor_multiplication(self, cpt_set: List[pd.DataFrame]) -> pd.DataFrame:
        """Input: a list of cpts 
           Output: result of factor moltiplication between the cpts"""

        var = []
        insts = []
        good_insts = []
        good_vars = []

        for cpt in cpt_set:
            for col_head in list(cpt.columns)[:-1]:
                if col_head not in var:
                    var.append(col_head)

        worlds = [list(i) for i in itertools.product([True, False], repeat=len(var))]

        for world in worlds:
            evi = {}
            for i, j in enumerate(var):
                evi[j] = world[i]
            insts.append(pd.Series(evi))
        
        for inst in insts:
            s = 0 #score
            for cpt in cpt_set:
                if not BayesNet.get_compatible_instantiations_table(inst, cpt).empty:
                    s += 1
            if s == len(cpt_set):
                good_insts.append(inst)
                
        list_vars = [list(i.to_dict().keys()) for i in good_insts]
        
        for i in list_vars:
            for j in i:
                if j not in good_vars:
                    good_vars.append(j)

        good_insts_val = [list(i.to_dict().values()) for i in good_insts]
        good_insts_val.sort()
        good_insts_val = list(good_insts for good_insts, _ in
                          itertools.groupby(good_insts_val)) 
        cpt_results = pd.DataFrame(good_insts_val, columns=good_vars)
        cpt_results['p'] = 1

        for i in range(len(good_insts_val)):
            inst = pd.Series(cpt_results.iloc[i][:-1], good_vars)
            for current_cpt in cpt_set:
                right_row = BayesNet.get_compatible_instantiations_table(inst, current_cpt)
                cpt_results.loc[i, 'p'] *= right_row['p'].values[0]
        return cpt_results

    def factor_multiplication_many(self, factors: List[pd.DataFrame]) -> pd.DataFrame:
        if len(factors) == 1:
            return factors[0]
        else:
            result = factors[0]
            for i in range(1, len(factors)):
                c_vars = list(result.columns.intersection(factors[i].columns)) #common variables
                if 'p' in c_vars:
                    c_vars.remove('p')
                if "archive" in c_vars:
                    c_vars.remove("archive")
                if len(c_vars) > 0:
                    result = self.multiply_factors(result, factors[i], c_vars)
            return result
    
    def multiply_factors(self, f1: pd.DataFrame, f2: pd.DataFrame, c_vars: List[str]) -> pd.DataFrame:
        merged = f1.merge(f2, on=c_vars, how='inner')
        merged = merged.assign(p=merged.p_x * merged.p_y, ).drop(columns=['p_x', 'p_y'])
        if "archive" in f1.columns and "archive" in f2.columns:
            merged = merged.assign(archive=merged.archive_x + merged.archive_y).drop(
                columns=['archive_x', 'archive_y'])
        return merged


    def min_deg(self, network: BayesNet, vars):
        """Takes a set of variables in the Bayesian Network 
        and eliminates X based on min-degree heuristics"""
        results = []
        int_graph = network.get_interaction_graph()
        while len(vars) > 0:
            s = [0 for _ in vars]  #scores
            for i in range(len(vars)):
                s[i] = int_graph.degree(vars[i])
            min_node = vars[np.argmin(s)]  
        #neighbour connections
            for n in int_graph.neighbors(min_node):  
                for k in int_graph.neighbors(min_node):
                    if k != n:
                        int_graph.add_edge(k, n)
        #removal    
            int_graph.remove_node(min_node)  
            results.append(min_node)  
            vars.remove(min_node)  
        return results

    def min_fill(self, network: BayesNet, vars):
        """Takes a set of variables in the Bayesian Network 
        and eliminates X based on min-fill heuristics"""
        results = []
        int_graph = network.get_interaction_graph()
        while len(vars) > 0:
            scores = [0 for i in vars]
            for i in range(len(vars)):
                """The score was calculated as a number of possible connection among neighbours (x_connections) 
                and number of connection that are actually present (connections)"""
                connections = 0
            #connections that could be possible
                x_connections = len(list(int_graph.neighbors(vars[i]))) * ( 
                        len(list(int_graph.neighbors(vars[i]))) - 1) / 2
            # neighbour connections
                for i in int_graph.neighbors(vars[i]): 
                    for j in int_graph.neighbors(vars[i]):
                        if int_graph.has_edge(i, j):
                            connections += 1
                scores[i] = x_connections - connections
            min_node = vars[np.argmin(scores)]
            for i in int_graph.neighbors(min_node):  
                for j in int_graph.neighbors(min_node):
                    if j != i:
                        int_graph.add_edge(j, i)
        #removal
            int_graph.remove_node(min_node)  
            results.append(min_node)
            vars.remove(min_node)  
        return results


    def marginal_distribution(self, Q: str, E: dict, elimination_order: list):

        for var in elimination_order:

            if var in E:

                boolean = E[var]

                cpt = self.bn.get_cpt(var)

                row_t = cpt.loc[cpt[var] == boolean]

                # Access the value in column 'p' of the retrieved row
                value_t = row_t.iloc[0]['p']

                # Convert the value to a float
                prob_t = pd.to_numeric(value_t, errors='coerce')

                var_children = self.bn.get_children(var)

                for child in var_children:
                    # get all the cpt's of the children
                    cpt_child = self.bn.get_cpt(child)

                    # Multiply the values in column 'p' where the corresponding value in column var is True by the float value
                    cpt_child.loc[cpt_child[var] == boolean, 'p'] = cpt_child.loc[cpt_child[var] == boolean, 'p'] * prob_t

                    # add a condition where the rows that are not in the evidence are deleted
                    cpt_child = cpt_child.drop(cpt_child[cpt_child[var] != boolean].index)

                    groups = cpt_child.columns.drop(var).drop('p').tolist()

                    # add together the rows with the same values
                    cpt_child = cpt_child.groupby(groups).sum().reset_index().drop(columns=[var])

                    # update the CPT of the variable
                    self.bn.update_cpt(child, cpt_child)

            else:
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

        cpt = self.bn.get_cpt(Q)
        query = Q

        row_t = cpt.loc[cpt[query] == True]
        row_f = cpt.loc[cpt[query] == False]

        # Access the value in column 'p' of the retrieved row
        value_t = row_t.iloc[0]['p']
        value_f = row_f.iloc[0]['p']

        # Convert the value to a float
        prob_t = pd.to_numeric(value_t, errors='coerce')
        prob_f = pd.to_numeric(value_f, errors='coerce')

        product_evidence = 0

        for key in E:
            value = E[key]
            cpt = self.bn.get_cpt(key)
            row_t = cpt.loc[cpt[key] == value]
            new_value = row_t.iloc[0]['p']
            float = pd.to_numeric(new_value, errors='coerce')
            product_evidence += float


        # Posterior marginal that Q is True given evidence E
        q_true = prob_t/product_evidence

        # Posterior marginal that Q is True given evidence E
        q_false = prob_f/product_evidence

        return q_true, q_false


reasoner = BNReasoner('use_case.BIFXML')

#print(reasoner.variable_elimination(['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?', 'Slippery Road?'], 'Slippery Road?'))
#print(reasoner.marginal_distribution('Sprinkler?', {'Winter?': True, 'Rain?': False}, ['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?', 'Slippery Road?']))

for var in reasoner.bn.get_all_variables():
    print(reasoner.bn.get_cpt(var))

reasoner.bn.get_interaction_graph()

# TODO: This is where your methods should go
