import copy
import itertools
from random import random
from typing import Union, List

import numpy as np
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

    def independent(self, X:[str], Y:[str], Z:[str]):
        return self.d_separation(X,Y,Z)


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

    def max_out(self, cpt: pd.DataFrame, var: str) -> pd.DataFrame:
        """Creates a new cpt with an additional column: memory
        memory keeps track of which instantiation of X led to maximized value.
        It is saved as a list of touples
        """
        
        if [var] == [col for col in cpt.columns if col != 'p' and col != 'memory']:
            print("ERROR: variable cannot be maxed out")
            exit()
            return None

        if var not in cpt.columns:
            print("ERROR: variable is not in cpt")
            return None

        excl_list = [var, 'p', 'memory']
        group_var = [col for col in cpt.columns if col not in excl_list and not col.startswith('memory')]
        cpt_new = cpt.loc[cpt.groupby(group_var)['p'].idxmax()]
        
        if "memory" not in cpt_new.columns:  
            list_dict = []
            for i, row in cpt_new.iterrows():
                list_dict.append([(var, row[var])])
            cpt_new.insert(0, "memory", list_dict)
        else:
            for i, row in cpt_new.iterrows():           
                memory_list = row["memory"]
                memory_list.append((var, row[var]))
                row["memory"] = memory_list
        cpt_new = cpt_new.drop(var, 1)
        return cpt_new
    
    
    
    def marginalization(self, cpt: pd.DataFrame, var_sum: str) -> pd.DataFrame:
        """Creates a new cpt without the variable var and probabilities after sum"""
        if [var_sum] == [col for col in cpt.columns if col != 'p' and col != 'memory']:
            print("ERROR : variable cannot be summed out")
            exit()
            return None
        var_new = [var for var in cpt.columns if var != var_sum]
        cpt_new = cpt[var_new].copy()
        # print("group by criteria:",[x for x in new_var if x != 'p'])
        group_vars = [var for var in var_new if var != 'p']
        res = pd.DataFrame(cpt_new.groupby(group_vars)['p'].sum()).reset_index()
        return res

    
    
    def in_cpt(self, cpt: pd.DataFrame, V: str) -> bool:
        """True if V is a column of cpt"""
        if [var for var in cpt.columns if var == V]:
            return True
        else:
            return False
    

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

    
    
    def multiply_factors(self, f1: pd.DataFrame, f2: pd.DataFrame, c_vars: List[str]) -> pd.DataFrame:
        merged = f1.merge(f2, on=c_vars, how='inner')
        merged = merged.assign(p=merged.p_x * merged.p_y, ).drop(columns=['p_x', 'p_y'])
        if "archive" in f1.columns and "archive" in f2.columns:
            merged = merged.assign(archive=merged.archive_x + merged.archive_y).drop(
                columns=['archive_x', 'archive_y'])
        return merged
    
    
    
    def factor_multiplication_many(self, factors: List[pd.DataFrame]) -> pd.DataFrame:
        if len(factors) == 1:
            return factors[0]
        else:
            result = factors[0]
            for i in range(1, len(factors)):
                c_vars = list(result.columns.intersection(factors[i].columns)) #common variables
                if 'p' in c_vars:
                    c_vars.remove('p')
                if "memory" in c_vars:
                    c_vars.remove("memory")
                if len(c_vars) > 0:
                    result = self.multiply_factors(result, factors[i], c_vars)
            return result
    
    
    
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
                """Scores are calculated as the number of connection that 
                are possible and the actual connection among neighbours"""
                connections = 0

            #connections that could be possible
                x_connections = len(list(int_graph.neighbors(vars[i]))) * ( 
                    len(list(int_graph.neighbors(vars[i]))) - 1) / 2

            # neighbour connections
                for j in int_graph.neighbors(vars[i]):
                    for h in int_graph.neighbors(vars[i]):
                        if int_graph.has_edge(j, h):
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

    def MAP(self, queries: List[str], e: pd.Series, heuristic: str, pruning: bool) -> dict:
        """Computes maximum a-posteriority instanstiation and value of variables Q, given evidence e (possibly empty)"""
        net = self.bn
        if pruning:
            self.prune(queries, list(e.keys()))
        not_queries = [i for i in net.get_all_variables() if i not in queries]
        if heuristic == "fill":
            ord = self.min_fill(net, not_queries)
        elif heuristic == "deg":
            ord = self.min_deg(net, not_queries)

        CIT = [net.get_compatible_instantiations_table(e, c).reset_index(drop=True) for c in
             net.get_all_cpts().values()]
        ord.extend(queries)
        results_dict = {}
        # print("Determining MAP for: {}, assuming: {}".format(queries, e.to_dict()))
        # print("Order: {}".format(ord))
        for var in ord:
            imp_factors = [f.reset_index(drop=True) for f in CIT if self.in_cpt(f, var)]
            t_factor = self.factor_multiplication_many(imp_factors)
            if len(t_factor.columns) <= 3 and "memory" in t_factor.columns:  
                results_idx = t_factor['p'].idxmax()
                results_dict[var] = t_factor.at[results_idx, var]
                for t in t_factor.at[results_idx, "memory"]:
                    results_dict[t[0]] = t[1]
                continue
            if len(t_factor.columns) <= 2:  
                if var in queries:
                    results_idx = t_factor['p'].idxmax()

                    results_dict[var] = t_factor.at[results_idx, var]
                    CIT = [i.reset_index(drop=True) for i in CIT if not self.in_cpt(i, var)]
                    continue
                else:
                    CIT = [i.reset_index(drop=True) for i in CIT if not self.in_cpt(i, var)]
                    continue
            if var not in queries:
                t_factor = self.marginalization(t_factor, var)
                CIT = [i.reset_index(drop=True) for i in CIT if not self.in_cpt(i, var)]
                CIT.append(t_factor)
            else:
                t_factor = self.max_out(t_factor, var)
                CIT = [i.reset_index(drop=True) for i in CIT if not self.in_cpt(i, var)]
                CIT.append(t_factor)
        return results_dict
    
    def MPE(self, e: pd.Series, heuristic: str, pruning: bool) -> dict:
        queries = list(filter(lambda var: var not in e.keys(), self.bn.get_all_variables()))
        return self.MAP(queries, e, heuristic, pruning)


reasoner = BNReasoner('testing/use_case.BIFXML')
print(reasoner.bn.get_all_variables())

for var in reasoner.bn.get_all_variables():
    print(reasoner.bn.get_cpt(var))

mar = reasoner.marginal_distribution('Feeling_happy', {'Work_on_assignment': True, 'Study_exam': True}, [])
print(mar)


# # TODO: This is where your methods should go
