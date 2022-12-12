reasoner = BNReasoner('testing/lecture_example.BIFXML')

for var in reasoner.bn.get_all_variables():

    print('variable is')
    print(var)
    cpt = reasoner.bn.get_cpt(var)
    print(cpt)

    row_t = cpt.loc[cpt[var] == True]
    row_f = cpt.loc[cpt[var] == False]

    # Access the value in column 'p' of the retrieved row
    value_t = row_t.iloc[0]['p']
    value_f = row_f.iloc[0]['p']

    # Convert the value to a float
    prob_t = pd.to_numeric(value_t, errors='coerce')
    prob_f = pd.to_numeric(value_f, errors='coerce')

    print(prob_t)
    print(prob_f)

    var_children = reasoner.bn.get_children(var)
    print('var children')
    print(var_children)

    for child in var_children:

        # get all the cpt's of the children
        cpt_child = reasoner.bn.get_cpt(child)
        print('cpt before')
        print(cpt_child)

        # Multiply the values in column 'p' where the corresponding value in column var is True by the float value
        cpt_child.loc[cpt_child[var] == True, 'p'] = cpt_child.loc[cpt_child[var] == True, 'p'] * prob_t

        # Multiply the values in column 'p' where the corresponding value in column var is False by the float value
        cpt_child.loc[cpt_child[var] == False, 'p'] = cpt_child.loc[cpt_child[var] == False, 'p'] * prob_f

        groups = cpt_child.columns.drop(var).drop('p').tolist()
        print(groups)

        # add together the rows with the same values
        cpt_child = cpt_child.groupby(groups).sum().reset_index().drop(columns=[var])

        # update the CPT of the variable
        #self.bn.update_cpt(child, cpt_child)
        reasoner.bn.update_cpt(child, cpt_child)

        cpt_2 = reasoner.bn.get_cpt(child)
        print(cpt_2)

        print('break')