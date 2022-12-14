import itertools
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from BNReasoner import BNReasoner

import numpy as np
import matplotlib.pyplot as plt


def p(variables):
    return [[o[:i], o[i:]] for o in list(itertools.permutations(variables)) for i in range(len(o))]

def time_sink():
    seconds = abs(np.random.normal(loc=0.0001, scale=0.00005, size=1)[0])
    time.sleep(seconds)

def all_evidence_combos(variables: [str]):
    all_combos = []
    [all_combos.extend(itertools.combinations(variables, i+1)) for i in range(len(variables))]
    all_combos = list(map(list, all_combos))
    return all_combos


def solve_with_pruning(query: [str], evidence: dict, heuristic:str, reasoner: BNReasoner):
    h = None
    if heuristic == 'min-fill':
        h = reasoner.min_fill
    elif heuristic == 'min-deg':
        h = reasoner.min_deg

    time_sink()
    #
    # evidence = {}
    #
    # reasoner.mpe_pruning(query, evidence, h)



    #reasoner.mpe_nonpruning(query, evidence, h)


def run_experiment(experiment_name, pruning:bool, heuristic: str, iterations: int):
    runtimes = []
    default_reasoner = BNReasoner('testing/use_case.BIFXML')
    all_combos = all_evidence_combos(default_reasoner.bn.get_all_variables())
    count = 0
    for evidence_combo in all_combos:
        evidence_dict = {}
        for var in evidence_combo:
            evidence_dict[var] = True

        evidence = pd.Series(evidence_dict)
        count = count + 1
        print("EVIDENCE SET: ", count)
        for _ in range(iterations):
            reasoner = deepcopy(default_reasoner)
            start = datetime.now()
            reasoner.MPE(reasoner.bn, evidence, heuristic, pruning)
            runtimes.append((datetime.now() - start).total_seconds())

    plt.boxplot(runtimes)
    plt.savefig(fname="experiments/"+experiment_name)

    result = {
            "avg": np.average(runtimes),
            "std": np.std(runtimes),
            "raw_data": runtimes}

    with open(Path("experiments").joinpath(Path(experiment_name).with_suffix(".json")), 'w') as outfile:
        json.dump(result, outfile, indent=4)

    return result


if __name__ == '__main__':
    run_experiment("PRUNE_FILL",pruning=True, heuristic="fill", iterations=100)
    run_experiment("NO_PRUNE_FILL",pruning=False, heuristic="fill", iterations=100)
    run_experiment("NO_PRUNE_DEG",pruning=False, heuristic="deg", iterations=100)
    run_experiment("PRUNE_DEG",pruning=True, heuristic="deg", iterations=100)
