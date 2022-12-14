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

def generate_query_evidence_pairs(variables: [str]):
    all_orders = list(itertools.permutations(variables))
    all_partitions = [{"query": tuple(sorted(o[:i])), "evidence": list(o[i:])} for o in all_orders for i in range(len(o))]

    final_result = pd.DataFrame(all_partitions).groupby("query").agg('first')
    return final_result


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


def solve_without_pruning(query: [str], evidence: dict, heuristic:str, reasoner: BNReasoner):
    h = None
    if heuristic == 'min-fill':
        h = reasoner.min_fill
    elif heuristic == 'min-deg':
        h = reasoner.min_deg

    evidence = {}
    time_sink()

    #reasoner.mpe_nonpruning(query, evidence, h)


def run_experiment(experiment_name, solver, heuristic: str, iterations: int):
    runtimes = []
    default_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    all_partitions = generate_query_evidence_pairs(default_reasoner.bn.get_all_variables())
    count = 0
    for query, row in all_partitions.iterrows():
        count = count + 1
        print("PARTITION: ", count)
        query = list(query)
        evidence = {}
        for var in row['evidence']:
            evidence[var] = True

        for _ in range(iterations):
            reasoner = deepcopy(default_reasoner)
            start = datetime.now()
            solver(query, evidence, heuristic, reasoner)
            runtimes.append((datetime.now() - start).microseconds)

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
    run_experiment("TEST", solver=solve_with_pruning, heuristic="min-deg", iterations=100)
